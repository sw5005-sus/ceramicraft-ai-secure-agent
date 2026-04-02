from confluent_kafka import Consumer, KafkaError, KafkaException
from ceramicraft_ai_secure_agent.kafka import order_handler, user_register_handler
from ceramicraft_ai_secure_agent.utils.logger import get_logger
from ceramicraft_ai_secure_agent.config.config import Config

logger = get_logger(__name__)


order_created_topic = "order_created"
user_activated_topic = "user_activated_topic"

topics = [order_created_topic, user_activated_topic]
topic2handler = {
    order_created_topic: order_handler,
    user_activated_topic: user_register_handler,
}


def create_consumer():
    logger.info("kafka consumer ready to consume")
    conf = {
        "bootstrap.servers": Config.kafka.bootstrap_servers,
        "group.id": Config.kafka.group_id,
        "auto.offset.reset": "earliest",
        "enable.auto.commit": True,
    }

    consumer = Consumer(conf)
    consumer.subscribe(topics)

    print(f"listen on Topic: {topics}...")

    try:
        while True:
            msg = consumer.poll(timeout=1.0)

            if msg is None:
                continue

            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    print(f"End of partition reached {msg.partition()}")
                else:
                    raise KafkaException(msg.error())
            else:
                key = msg.key().decode("utf-8") if msg.key() else None
                value = msg.value().decode("utf-8") if msg.value() else None
                logger.info(
                    f"Message received: Key={key}, Value={value}, Partition={msg.partition()}, Offset={msg.offset()}"
                )
                topic = msg.topic()
                if not topic2handler[topic]:
                    logger.warning(f"unkonwn topic {topic}, will ignore")
                    continue
                topic2handler[msg.topic()].handle(value)

    except KeyboardInterrupt:
        print("consume interrupted")
    finally:
        consumer.close()
