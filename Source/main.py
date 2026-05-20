# -*coding: utf-8 -*-
import JackFramework as jf
from UserModelImplementation import UserInterface


def main() -> None:
    app = jf.Application(UserInterface(), "Template Matching model")
    app.start()


# execute the main function
if __name__ == "__main__":
    main()
