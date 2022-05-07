from datetime import datetime, time, timedelta
from functools import lru_cache
from random import randrange
from typing import Iterator, T_co

import numpy as np
import xarray as xr
from numpy import float32
from torch.utils.data import IterableDataset

from pvlib import solarposition # for determining day times
import pandas as pd
from OSGridConverter import grid2latlong
from bng_latlon import OSGB36toWGS84
from numpy import float64

class ClimateHackDataset(IterableDataset):
    def __init__(
        self,
        dataset: xr.Dataset,
        start_date: datetime = None,
        end_date: datetime = None,
        crops_per_slice: int = 1,
        day_limit: int = 0,
    ) -> None:
        super().__init__()

        self.dataset = dataset
        self.crops_per_slice = crops_per_slice
        self.day_limit = day_limit
        self.cached_items = []
        
        self.coordinates = []
        self.features = []
        self.labels = []

        times = self.dataset.get_index("time")
        self.min_date = times[0].date()
        self.max_date = times[-1].date()

        if start_date is not None:
            self.min_date = max(self.min_date, start_date)

        if end_date is not None:
            self.max_date = min(self.max_date, end_date)
        elif self.day_limit > 0:
            self.max_date = min(
                self.max_date, self.min_date + timedelta(days=self.day_limit)
            )
    def _image_times(self, start_time, end_time):
        date = self.min_date
        while date <= self.max_date:
            current_time = datetime.combine(date, start_time)
            while current_time.time() <= end_time:
                yield current_time
                current_time += timedelta(minutes=5) #originally 20

            date += timedelta(days=1)
            
    def _get_crop(self, input_slice, target_slice):
        # roughly over the mainland UK
        rand_x = randrange(550, 950 - 96)
        rand_y = randrange(375, 700 - 96)

        # make a data selection
        selection = input_slice.isel(
            x=slice(rand_x, rand_x + 96),
            y=slice(rand_y, rand_y + 96),
        )
        ##time 
#         time = selection["time"].values
        # get the OSGB coordinate data
        osgb_data = np.stack(
            [
                selection["x_osgb"].values.astype(float32),
                selection["y_osgb"].values.astype(float32),
            ]
        )

        if osgb_data.shape != (2, 96, 96):
            return None
        
        # get the input satellite imagery
        input_data = selection["data"].values.astype(float32)
        if input_data.shape != (1, 96, 96):
            return None

        # get the target output
        target_output = (
            target_slice["data"]
            .isel(
                x=slice(rand_x, rand_x + 96),
                y=slice(rand_y, rand_y + 96),
            )
            .values.astype(float32)   
        )
        #solar positioning
        target = target_slice.isel(
            x=slice(rand_x, rand_x + 96),
            y=slice(rand_y, rand_y + 96),
        )
        intime = pd.DatetimeIndex(selection["time"].values)
        tartime = pd.DatetimeIndex(target["time"].values) 
        
#         check that center is daylight
        inlat, inlon = OSGB36toWGS84(selection.x_osgb[48][48].values, selection.y_osgb[48][48].values)
        solpos = solarposition.get_solarposition(intime, inlat, inlon)
        if solpos['apparent_elevation'][0] < 10:
            return None
        tarlat, tarlon = OSGB36toWGS84(target.x_osgb[48][48].values, target.y_osgb[48][48].values)
        tarsolpos = solarposition.get_solarposition(tartime, tarlat, tarlon)
        if tarsolpos['apparent_elevation'][0] < 10:
            return None
        
        
        inlat, inlon = OSGB36toWGS84(selection.x_osgb[0][48].values, selection.y_osgb[0][48].values)
        solpos = solarposition.get_solarposition(intime, inlat, inlon)
        if solpos['apparent_elevation'][0] < 10:
            return None
        inlat, inlon = OSGB36toWGS84(selection.x_osgb[48][0].values, selection.y_osgb[48][0].values)
        solpos = solarposition.get_solarposition(intime, inlat, inlon)
        if solpos['apparent_elevation'][0] < 10:
            return None
        inlat, inlon = OSGB36toWGS84(selection.x_osgb[95][48].values, selection.y_osgb[95][48].values)
        solpos = solarposition.get_solarposition(intime, inlat, inlon)
        if solpos['apparent_elevation'][0] < 10:
            return None
        inlat, inlon = OSGB36toWGS84(selection.x_osgb[48][95].values, selection.y_osgb[48][95].values)
        solpos = solarposition.get_solarposition(intime, inlat, inlon)
        if solpos['apparent_elevation'][0] < 10:
            return None
        inlat, inlon = OSGB36toWGS84(selection.x_osgb[0][0].values, selection.y_osgb[0][0].values)
        solpos = solarposition.get_solarposition(intime, inlat, inlon)
        if solpos['apparent_elevation'][0] < 10:
            return None
        inlat, inlon = OSGB36toWGS84(selection.x_osgb[0][95].values, selection.y_osgb[0][95].values)
        solpos = solarposition.get_solarposition(intime, inlat, inlon)
        if solpos['apparent_elevation'][0] < 10:
            return None
        inlat, inlon = OSGB36toWGS84(selection.x_osgb[95][95].values, selection.y_osgb[95][95].values)
        solpos = solarposition.get_solarposition(intime, inlat, inlon)
        if solpos['apparent_elevation'][0] < 10:
            return None
        inlat, inlon = OSGB36toWGS84(selection.x_osgb[95][0].values, selection.y_osgb[95][0].values)
        solpos = solarposition.get_solarposition(intime, inlat, inlon)
        if solpos['apparent_elevation'][0] < 10:
            return None
        
        tarlat, tarlon = OSGB36toWGS84(target.x_osgb[0][48].values, target.y_osgb[0][48].values)
        tarsolpos = solarposition.get_solarposition(tartime, tarlat, tarlon)
        if tarsolpos['apparent_elevation'][0] < 10:
            return None
        tarlat, tarlon = OSGB36toWGS84(target.x_osgb[48][0].values, target.y_osgb[48][0].values)
        tarsolpos = solarposition.get_solarposition(tartime, tarlat, tarlon)
        if tarsolpos['apparent_elevation'][0] < 10:
            return None
        tarlat, tarlon = OSGB36toWGS84(target.x_osgb[95][48].values, target.y_osgb[95][48].values)
        tarsolpos = solarposition.get_solarposition(tartime, tarlat, tarlon)
        if tarsolpos['apparent_elevation'][0] < 10:
            return None
        tarlat, tarlon = OSGB36toWGS84(target.x_osgb[48][95].values, target.y_osgb[48][95].values)
        tarsolpos = solarposition.get_solarposition(tartime, tarlat, tarlon)
        if tarsolpos['apparent_elevation'][0] < 10:
            return None
        tarlat, tarlon = OSGB36toWGS84(target.x_osgb[0][0].values, target.y_osgb[0][0].values)
        tarsolpos = solarposition.get_solarposition(tartime, tarlat, tarlon)
        if tarsolpos['apparent_elevation'][0] < 10:
            return None
        tarlat, tarlon = OSGB36toWGS84(target.x_osgb[0][95].values, target.y_osgb[0][95].values)
        tarsolpos = solarposition.get_solarposition(tartime, tarlat, tarlon)
        if tarsolpos['apparent_elevation'][0] < 10:
            return None
        tarlat, tarlon = OSGB36toWGS84(target.x_osgb[95][95].values, target.y_osgb[95][95].values)
        tarsolpos = solarposition.get_solarposition(tartime, tarlat, tarlon)
        if tarsolpos['apparent_elevation'][0] < 10:
            return None
        tarlat, tarlon = OSGB36toWGS84(target.x_osgb[95][0].values, target.y_osgb[95][0].values)
        tarsolpos = solarposition.get_solarposition(tartime, tarlat, tarlon)
        if tarsolpos['apparent_elevation'][0] < 10:
            return None
        
        if target_output.shape != (1, 96, 96):
            return None

        return osgb_data, input_data, target_output

    def __iter__(self) -> Iterator[T_co]:
        if self.cached_items:
            for item in self.cached_items:
                yield item

            return

        start_time = time(5, 0)
        end_time = time(18, 0)

        for current_time in self._image_times(start_time, end_time):
            data_slice = self.dataset.loc[
                {
                    "time": slice(
                        current_time,
                        current_time + timedelta(hours=0, minutes=5),
                    )
                }
            ]

            if data_slice.sizes["time"] != 2:
                continue

            input_slice = data_slice.isel(time=slice(0, 1))
            target_slice = data_slice.isel(time=slice(1, 2))

            crops = 0
            while crops < self.crops_per_slice:
                crop = self._get_crop(input_slice, target_slice)
                if crop:
                    self.cached_items.append(crop)
                    yield crop

                crops += 1