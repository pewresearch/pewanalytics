from builtins import object
import re
import datetime

from dateutil.parser import parse
from calendar import IllegalMonthError


class DateFinder(object):

    """
    A helper class to search for dates in text using a series of regular expressions and a parser from \
    :py:mod:`dateutil`. Verifies that :py:mod:`dateutil` did not auto-fill missing values in the date. Time \
    information will be automatically cleared out, but you can also pass a list of additional regular expression \
    patterns (as strings) to define other patterns that should be cleared out before scanning for dates.

    :param preprocessing_patterns: Optional list of additional patterns to clear out prior to searching for dates.
    :type preprocessing_patterns: list

    Usage::

        from pewanalytics.text.dates import DateFinder

        text = "January 1, 2018 and 02/01/2019 and Mar. 1st 2020"
        low_bound = datetime.datetime(2017, 1, 1)
        high_bound = datetime.datetime(2021, 1, 1)

        >>> finder = DateFinder()
        >>> dates = finder.find_dates(text, low_bound, high_bound)
        >>> dates
        [
            (datetime.datetime(2018, 1, 1, 0, 0), 'January 1, 2018 '),
            (datetime.datetime(2020, 3, 1, 0, 0), 'Mar. 1st 2020'),
            (datetime.datetime(2019, 2, 1, 0, 0), '02/01/2019 ')
        ]

    """

    def __init__(self, preprocessing_patterns=None):

        # A generally permissive date regex, also fairly prone to false positives
        self.date_regex = re.compile(
            r"""(?=((?ix)                       # case-insensitive, verbose regex
            \b                                  # match a word boundary
            (?:                                 # match the following three times:
             (?:                                # either
              \d+                               # a number,
              (?:\.|st|nd|rd|th|,)*             # followed by a dot, st, nd, rd, a comma, or th (optional)
              |                                 # or a month name
              (?:(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*)
             )
             [\s./-]*                           # followed by a date separator or whitespace (optional)
            ){3}                                # do this three times
            \b)) """
        )

        # Always remove times, because those trip up dates
        self.preprocessing_patterns = [re.compile("((?:\d\d|\d):[0-9][0-9])")]
        # Add in any additional patterns to clear out, as provided by the user
        if preprocessing_patterns:
            self.preprocessing_patterns.extend(
                [re.compile(p) for p in preprocessing_patterns]
            )

    def _preprocess(self, text):

        """
        Return the text without any references to "parts" or specific times.

        :param text: A string to be cleaned that contains a date
        :type text: str
        :return: A cleaned string without additional time info and other boilerplate
        :rtype: str
        """

        for pattern in self.preprocessing_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                text = text.replace(match, "")

        return text

    def find_dates(self, text, cutoff_date_start, cutoff_date_end):

        """
        Return all of the dates (in text form and as datetime) in the text variable that fall within the specified \
        window of dates (inclusive).

        :param text: The text to scan for dates
        :type text: str
        :param cutoff_date_start: No dates will be returned if they fall before this date
        :type cutoff_date_start: `datetime.date`
        :param cutoff_date_end: No dates will be returned if they fall after this date
        :type cutoff_date_end: `datetime.date`
        :return: A list of tuples containing (datetime object, raw date text)
        :rtype: list
        """

        final_dates, suspected_dates = [], []

        # Start by stripping out time info, which can confuse the date regex and will lead to text snippets that \
        # can't be parsed into dates.
        text = self._preprocess(text)

        # Now find all of the plausible dates in the text. Many of these will be false positives.
        date_text_list = re.findall(self.date_regex, text)
        for date_raw_text in date_text_list:
            suspected_dates.append(date_raw_text)

        for date_raw_text in suspected_dates:

            # The dateutil parser fills in missing date components with the current day's date, or with a default \
            # date you can pass it. This is annoying, because partial dates could show up in bulk, leading to odd \
            # clumps of dates on the same date that you parsed. Our solution is to use the two cutoff dates as default \
            # dates, parsing each date twice. If the two dates are the same, then the default date was not used. \
            # The contents of these dates don't matter much. However, they cannot have a month, day OR year in common, \
            # otherwise this won't work.
            default_date_1 = datetime.datetime(day=1, year=2020, month=9)
            default_date_2 = datetime.datetime(day=2, year=1999, month=6)

            try:
                datetime_1 = parse(date_raw_text, fuzzy=True, default=default_date_1)
                datetime_2 = parse(date_raw_text, fuzzy=True, default=default_date_2)
                if (
                    datetime_1 == datetime_2
                    and cutoff_date_start <= datetime_1 <= cutoff_date_end
                ):
                    date = (datetime_1, date_raw_text)
                    final_dates.append(date)
            except (ValueError, IllegalMonthError):
                pass

        return list(set(final_dates))
