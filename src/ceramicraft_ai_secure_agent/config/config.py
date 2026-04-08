import os
import threading

import yaml
from pydantic import BaseModel


class RedisConfig(BaseModel):
    host: str
    port: int


class KafkaConfig(BaseModel):
    bootstrap_servers: str
    group_id: str


class MysqlConfig(BaseModel):
    host: str
    port: int
    username: str
    database: str
    max_connect_size: int


class HttpServerConfig(BaseModel):
    host: str
    port: int


class Config(BaseModel):
    http: HttpServerConfig
    redis: RedisConfig
    kafka: KafkaConfig
    mysql: MysqlConfig


def load_config() -> Config:
    config_path = None
    if "CONFIG_PATH" in os.environ:
        config_path = os.environ["CONFIG_PATH"]
        print(
            f"Using config path from environment variable CONFIG_PATH: {config_path}",
            flush=True,
        )
    else:
        current_file_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_file_dir, "config.yaml")
        print(f"CONFIG_PATH not set, using default path: {config_path}", flush=True)
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
    return Config(**config_dict)


_config_lock = threading.Lock()
_system_config: Config | None = None


def get_config() -> Config:
    """Thread-safe singleton getter for the system configuration."""
    global _system_config
    if _system_config is None:
        with _config_lock:
            if _system_config is None:  # Double-checked locking
                _system_config = load_config()
    return _system_config
