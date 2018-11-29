from builtins import object
import re
import datetime
import numpy as np

from dateutil.parser import parse
from calendar import IllegalMonthError


class DateFinder(object):
    """Search bodies of text for dates""" 
    def __init__(self):

        # A generally permissive date regex, also fairly prone to false positives
        self.general_date_regex = self._compile_permissive_date_regex()

        # A regex to identify reference to sermon "parts" (part 1...), which we remove because those trip up the dates.
        self.part_regex = self._compile_part_regex()

        # Removing times, because those also trip up dates
        self.standard_time_regex = self._compile_time_regex()

    def _compile_part_regex(self):
        """:output: regex that breaks dates into component parts"""
        part_regex = re.compile(
            r"""(?ix)             # case-insensitive, verbose regex
            \b                    # match a word boundary
            (?:(?:part|pt\.|pt|p\.)[a-z]*)
             [\s]+                 # followed by whitespace
             \d+                   # Then a number,
            \b """)

        return part_regex


    def _compile_permissive_date_regex(self):
        """:output: regex looking for dates """
        general_date_regex = re.compile(
            r"""(?=((?ix)             # case-insensitive, verbose regex
            \b                    # match a word boundary
            (?:                   # match the following three times:
             (?:                  # either
              \d+                 # a number,
              (?:\.|st|nd|rd|th|,)* # followed by a dot, st, nd, rd, a comma, or th (optional)
              |                   # or a month name
              (?:(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*)
             )
             [\s./-]*             # followed by a date separator or whitespace (optional)
            ){3}                  # do this three times
            \b)) """)

        return general_date_regex


    def _compile_time_regex(self):
        """Return a regex that searches for times"""
        standard_time_regex = re.compile("((?:\d\d|\d):[0-9][0-9])")
        return standard_time_regex


    def _remove_part_and_time_references(self, text):
        """Return the text without any references to "parts" or specific times.
            :param text:
        """
        part_references = re.findall(self.part_regex, text)

        for reference in part_references:
            text = text.replace(reference, "")

        time_references = re.findall(self.standard_time_regex, text)

        for reference in time_references:
            text = text.replace(reference, "")

        return text


    def find_dates(self, text, cutoff_date_start, cutoff_date_end):
        """Return all of the dates (in text form and as datetime) in the text variable,
        but none before or after the start and end cutoff dates.

        :param text: to search
        :param cutoff_date_start: no dates will be found before this date
        :param cutoff_date_end: no dates will be returned after this date

        :output: dates list
        """
        final_dates, suspected_dates = [], []

        # Start by stripping out all references to parts, like "pt. 1" from the text, as well as times like "March 12 2018, 8:00. These
        # confuse the date regex, and will lead to text snippets that can't be parsed into dates. Worse, if the snippet containing "pt 1"
        # overlaps with a real date, those dates will be excluded because the regex is non-overlapping.
        text = self._remove_part_and_time_references(text)

        # Now find all of the plausible dates in the text. Many of these will be false positives.
        date_text_list = re.findall(self.general_date_regex, text)
        for date_raw_text in date_text_list:
            suspected_dates.append(date_raw_text)

        for date_raw_text in suspected_dates:
            # The date parser fills in missing date components with the current days' date, or with a default date you can pass it.
            # This is annoying, because partial dates could show up in bulk, leading to odd clumps of dates on the same date that you parsed.
            # Our solution is to use the two cutoff dates as default dates, parsing each date twice. If the two dates are the same, then
            # the default date was not used.

            # The contents of these dates don't matter much. However, they cannot have a month, day OR year in common, otherwise
            # this won't work.
            default_date_1 = datetime.datetime(day=1, year = 2020, month = 9)
            default_date_2 = datetime.datetime(day=2, year = 1999, month = 6)

            try:
                datetime_1 = parse(date_raw_text, fuzzy=True, default=default_date_1)
                datetime_2 = parse(date_raw_text, fuzzy=True, default=default_date_2)
                if datetime_1 == datetime_2 and  cutoff_date_start < datetime_1 < cutoff_date_end:
                    date = (datetime_1, date_raw_text)
                    final_dates.append(date)
            except (ValueError, IllegalMonthError):
                pass

        return list(set(final_dates))
