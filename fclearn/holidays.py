"""Module that extends the holidays library to include vacations and Carnaval."""

import datetime

import holidays


class DutchHolidays(holidays.NL):
    """Creates a dictionary with all the Dutch holidays."""

    def _populate(self, year):
        holidays.NL._populate(self, year)

        # Add carnaval
        easter = self.get_named("Eerste Paasdag")[0]
        carnaval_dates = [easter - datetime.timedelta(days=day) for day in [48, 47, 46]]
        for date in carnaval_dates:
            self[date] = "Carnaval"

        # Add 'bouwvak'
        # 'Bouwvak' starts at the last full week of July
        # (the week with the 26th of July) and rotates every year
        start_bv = datetime.date(year, 7, 26) - datetime.timedelta(
            days=datetime.date(year, 7, 26).weekday()
        )

        offset = [
            {"zuid": 0, "midden": 7, "noord": 14},
            {"zuid": 0, "noord": 7, "midden": 14},
            {"noord": 0, "zuid": 7, "midden": 14},
            {"noord": 0, "midden": 7, "zuid": 14},
            {"midden": 0, "noord": 7, "zuid": 14},
            {"midden": 0, "zuid": 7, "noord": 14},
        ]

        year_cycle = 2018 % 6

        for key, value in offset[year_cycle].items():
            bv_dates = [
                start_bv + datetime.timedelta(days=(day + value))
                for day in range(0, 21)
            ]

            for date in bv_dates:
                self[date] = "Bouwvak {}".format(key)

        # Merge Pasen, Kerst and Pinksteren into one event
        to_replace = {
            "Paasdag": "Pasen",
            "Kerstdag": "Kerst",
            "Pinksterdag": "Pinksteren",
        }
        to_add = []

        # Remove and rename
        for key, value in to_replace.items():
            dates = self.get_named(key)
            for date in dates:
                to_add.append((date, value))
            self.pop_named(key)

        # Add again
        for date, name in to_add:
            self[date] = name
