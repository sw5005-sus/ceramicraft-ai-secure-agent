#!/bin/bash

# Check if the correct number of arguments is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 {type}"
    exit 1
fi

TYPE=$1
MESSAGE_FILE_PATH=''
# Set the topic based on the type
if [ "$TYPE" == "user" ]; then
    TOPIC="user-activated"
    MESSAGE_FILE_PATH="user_activated.json"
elif [ "$TYPE" == "order" ]; then
    TOPIC="order_created"
    MESSAGE_FILE_PATH="order_created.json"
else
    echo "Invalid type. Use 'user' or 'order'."
    exit 1
fi

# Execute the command
uv run python kafka_producer.py --topic "$TOPIC" --file "$MESSAGE_FILE_PATH"