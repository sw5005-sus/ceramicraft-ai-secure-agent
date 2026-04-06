import argparse
import asyncio
import json

from aiokafka import AIOKafkaProducer


async def send_kafka_message(topic, file_path):
    producer = AIOKafkaProducer(
        bootstrap_servers="10.249.171.135:9092",
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    )

    await producer.start()

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        messages = data if isinstance(data, list) else [data]

        tasks = []
        for msg in messages:
            tasks.append(producer.send(topic, msg))

        await asyncio.gather(*tasks)
        print(f"\n[+] send {len(messages)} messages to Topic: {topic}")

    except FileNotFoundError:
        print(f"[-] error: file not found '{file_path}'")
    except json.JSONDecodeError:
        print(f"[-] error: file '{file_path}' is not a valid JSON")
    except Exception as e:
        print(f"[-] unknown error: {e}")
    finally:
        await producer.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="user aiokafka send JSON data")
    parser.add_argument("--topic", help="topic name to send messages to")
    parser.add_argument(
        "--file", help="json file path containing the message(s) to send"
    )

    args = parser.parse_args()

    asyncio.run(send_kafka_message(args.topic, args.file))
