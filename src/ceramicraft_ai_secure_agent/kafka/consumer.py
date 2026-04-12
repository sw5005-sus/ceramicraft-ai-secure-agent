import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Sequence

from aiokafka import AIOKafkaConsumer
from opentelemetry import context, metrics, propagate, trace
from opentelemetry.propagators.textmap import Getter
from opentelemetry.trace import SpanKind, Status, StatusCode

from ceramicraft_ai_secure_agent.config.config import get_config
from ceramicraft_ai_secure_agent.kafka import order_handler, user_register_handler
from ceramicraft_ai_secure_agent.utils.logger import get_logger

logger = get_logger(__name__)
tracer = trace.get_tracer(__name__)
meter = metrics.get_meter("ceramicraft_ai_secure_agent.kafka.consumer")

order_created_topic = "order_created"
user_activated_topic = "user-activated"

topics = [order_created_topic, user_activated_topic]
topic2handler = {
    order_created_topic: order_handler,
    user_activated_topic: user_register_handler,
}

executor = ThreadPoolExecutor(max_workers=10)

messages_total = meter.create_counter(
    "kafka_consumer_received_messages_total",
    description="Total number of Kafka messages consumed",
)

errors_total = meter.create_counter(
    "kafka_consumer_errors_total",
    description="Total number of Kafka message processing errors",
)

processing_duration_ms = meter.create_histogram(
    "kafka_consumer_processing_duration_ms",
    unit="ms",
    description="Kafka message processing duration in milliseconds",
)

message_lag_ms = meter.create_histogram(
    "kafka_consumer_message_lag_ms",
    unit="ms",
    description="Lag between Kafka message timestamp and consume time in milliseconds",
)


KafkaHeaderCarrier = Sequence[tuple[str, bytes]]


class KafkaHeadersGetter(Getter[KafkaHeaderCarrier]):
    def get(
        self, carrier: Optional[KafkaHeaderCarrier], key: str
    ) -> Optional[list[str]]:
        if carrier is None:
            return None

        key_lower = key.lower()
        values: list[str] = []

        for k, v in carrier:
            if k.lower() == key_lower:
                try:
                    values.append(v.decode("utf-8"))
                except Exception:
                    logger.exception(
                        f"Failed to decode Kafka header value for key '{k}'"
                    )

        return values or None

    def keys(self, carrier: Optional[KafkaHeaderCarrier]) -> list[str]:
        if carrier is None:
            return []
        return [k for k, _ in carrier]


headers_getter = KafkaHeadersGetter()


def _build_metric_attrs(
    msg, operation: str, handler_name: str | None = None, status: str | None = None
):
    attrs = {
        "messaging.system": "kafka",
        "messaging.destination.name": msg.topic,
        "messaging.kafka.partition": msg.partition,
        "messaging.operation": operation,
        "consumer.group.name": get_config().kafka.group_id,
    }
    if handler_name:
        attrs["messaging.consumer.handler"] = handler_name
    if status:
        attrs["messaging.process.status"] = status
    return attrs


def _run_handler_with_span(handler, value: str, msg, parent_ctx):
    handler_name = getattr(handler, "__name__", handler.__class__.__name__)

    start = time.perf_counter()
    token = context.attach(parent_ctx)
    try:
        with tracer.start_as_current_span(
            name=f"{msg.topic} process",
            kind=SpanKind.CONSUMER,
            attributes=_build_metric_attrs(
                msg, operation="process", handler_name=handler_name
            ),
        ) as span:
            try:
                handler.handle(value)
                span.set_status(Status(StatusCode.OK))
                processing_duration_ms.record(
                    (time.perf_counter() - start) * 1000,
                    _build_metric_attrs(
                        msg, operation="process", handler_name=handler_name, status="ok"
                    ),
                )
            except Exception as e:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
                errors_total.add(
                    1,
                    _build_metric_attrs(
                        msg,
                        operation="process",
                        handler_name=handler_name,
                        status="error",
                    ),
                )
                processing_duration_ms.record(
                    (time.perf_counter() - start) * 1000,
                    _build_metric_attrs(
                        msg,
                        operation="process",
                        handler_name=handler_name,
                        status="error",
                    ),
                )
                raise
    finally:
        context.detach(token)


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
    logger.info("listen on Topic: %s...", topics)

    try:
        async for msg in consumer:
            extracted_ctx = propagate.extract(msg.headers, getter=headers_getter)
            lag = max(0, int(time.time() * 1000) - msg.timestamp)

            with tracer.start_as_current_span(
                name=f"{msg.topic} receive",
                context=extracted_ctx,
                kind=SpanKind.CONSUMER,
                attributes=_build_metric_attrs(msg=msg, operation="receive"),
            ) as span:
                try:
                    key = msg.key.decode("utf-8") if msg.key else None
                    if msg.value is None:
                        logger.warning(
                            "Message value is empty, skipping. topic=%s offset=%s",
                            msg.topic,
                            msg.offset,
                        )
                        continue

                    value = msg.value.decode("utf-8")

                    span.set_attribute("messaging.kafka.message_key", key or "")
                    span.set_attribute(
                        "messaging.message.payload_size_bytes",
                        len(msg.value) if msg.value else 0,
                    )

                    logger.info(
                        "Message received: Key=%s, Topic=%s, "
                        "Partition=%s, Offset=%s, Message=%s",
                        key,
                        msg.topic,
                        msg.partition,
                        msg.offset,
                        value,
                    )

                    messages_total.add(
                        1,
                        _build_metric_attrs(
                            msg, operation="receive", status="received"
                        ),
                    )
                    message_lag_ms.record(
                        lag,
                        _build_metric_attrs(
                            msg, operation="receive", status="received"
                        ),
                    )

                    handler = topic2handler.get(msg.topic)
                    if not handler:
                        logger.warning("Unknown topic %s, ignoring", msg.topic)
                        span.set_attribute("messaging.process.status", "ignored")
                        continue

                    current_ctx = trace.set_span_in_context(span)
                    await loop.run_in_executor(
                        executor,
                        _run_handler_with_span,
                        handler,
                        value,
                        msg,
                        current_ctx,
                    )

                    span.set_status(Status(StatusCode.OK))
                except Exception as e:
                    span.record_exception(e)
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    logger.exception("Error in receive span")
    except Exception:
        logger.exception("Kafka consumer error")
    finally:
        await consumer.stop()
