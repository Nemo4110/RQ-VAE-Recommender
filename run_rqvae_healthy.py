"""Gin-safe command-line entry point for the isolated healthy RQ-VAE probe."""

from modules.utils import parse_config
from train_rqvae_healthy import train


if __name__ == "__main__":
    parse_config()
    train()
