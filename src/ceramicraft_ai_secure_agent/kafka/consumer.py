import asyncio
from concurrent.futures import ThreadPoolExecutor

from aiokafka import AIOKafkaConsumer

from ceramicraft_ai_secure_agent.config.config import get_config
from ceramicraft_ai_secure_agent.kafka import order_handler, user_register_handler
from ceramicraft_ai_secure_agent.utils.logger import get_logger

logger = get_logger(__name__)


order_created_topic = "order_created"
user_activated_topic = "user-activated"

topics = [order_created_topic, user_activated_topic]
topic2handler = {
    order_created_topic: order_handler,
    user_activated_topic: user_register_handler,
}

executor = ThreadPoolExecutor(max_workers=10)


async def consume():
    logger.info("kafka consumer ready to consume")
    loop = asyncio.get_running_loop()
    consumer = AIOKafkaConsumer(
        *topics,
        bootstrap_servers=get_config().kafka.bootstrap_servers,
        group_id=get_config().kafka.group_id,
        enable_auto_commit=True,
        session_timeout_ms=30000,
        heartbeat_interval_ms=10000,
        max_poll_interval_ms=300000,
    )
    await consumer.start()
    logger.info(f"listen on Topic: {topics}...")

    try:
        async for msg in consumer:
            try:
                key = msg.key.decode("utf-8") if msg.key else None
                value = msg.value.decode("utf-8") if msg.value else None

                logger.info(
                    f"Message received: Key={key}, Topic={msg.topic}, "
                    f"Partition={msg.partition}, Offset={msg.offset},"
                    f"Message={value}"
                )

                handler = topic2handler.get(msg.topic)
                if not handler:
                    logger.warning(f"Unknown topic {msg.topic}, ignoring")
                    continue

                await loop.run_in_executor(executor, handler.handle, value)

            except Exception as e:
                logger.error(f"Error processing message: {e}")

    except Exception as e:
        logger.error(f"Kafka consumer error: {e}")
    finally:
        await consumer.stop()
