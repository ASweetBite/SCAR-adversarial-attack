import logging
import os
from pathlib import Path

def setup_logger(log_file_path: str, log_level: int = logging.INFO) -> logging.Logger:
    """
    通用日志配置工具
    :param log_file_path: 日志保存路径（如 ./logs/main.log）
    :param log_level: 日志级别（默认 INFO）
    :return: 配置好的 logger 实例
    """
    # 自动创建日志目录
    log_dir = os.path.dirname(log_file_path)
    if log_dir and not os.path.exists(log_dir):
        Path(log_dir).mkdir(parents=True, exist_ok=True)

    # 定义日志格式（时间 - 级别 - 信息）
    log_format = "%(asctime)s | %(levelname)-7s | %(message)s"
    formatter = logging.Formatter(log_format, datefmt="%Y-%m-%d %H:%M:%S")

    # 获取 logger 对象（避免重复创建）
    logger = logging.getLogger()
    logger.setLevel(log_level)
    # 清空已有处理器（防止重复打印）
    logger.handlers.clear()

    # 1. 文件处理器：写入日志文件
    file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # 2. 控制台处理器：打印到终端
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger