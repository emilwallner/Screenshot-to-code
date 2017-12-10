__author__ = 'Tony Beltramelli - www.tonybeltramelli.com'

import string
import random


class Utils:
    @staticmethod
    def get_random_text(length_text=10, space_number=1, with_upper_case=True):
        results = []
        while len(results) < length_text:
            char = random.choice(string.ascii_letters[:26])
            results.append(char)
        if with_upper_case:
            results[0] = results[0].upper()

        current_spaces = []
        while len(current_spaces) < space_number:
            space_pos = random.randint(2, length_text - 3)
            if space_pos in current_spaces:
                break
            results[space_pos] = " "
            if with_upper_case:
                results[space_pos + 1] = results[space_pos - 1].upper()

            current_spaces.append(space_pos)

        return ''.join(results)

    @staticmethod
    def get_ios_id(length=10):
        results = []

        while len(results) < length:
            char = random.choice(string.digits + string.ascii_letters)
            results.append(char)

        results[3] = "-"
        results[6] = "-"

        return ''.join(results)

    @staticmethod
    def get_android_id(length=10):
        results = []

        while len(results) < length:
            char = random.choice(string.ascii_letters)
            results.append(char)

        return ''.join(results)
