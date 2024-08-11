# Copyright © 2024- Frello Technology Private Limited

from fire import Fire

from tuneapi import __version__


def main():

    Fire({"version": __version__})


if __name__ == "__main__":
    main()
