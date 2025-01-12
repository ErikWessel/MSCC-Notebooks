import datetime
import gc
import ipaddress
import logging
import math
import os
import pathlib
import shutil
import time
import warnings
from abc import ABC
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Tuple, Union
from zipfile import ZipFile

import geopandas as gpd
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import networkx
import networkx.convert
import numpy as np
import pandas as pd
import rasterio
import rasterio.features
import rasterio.mask
import rasterio.plot
import rasterio.warp
import seaborn as sns
import shapely
import shapely.geometry
import shapely.wkt
import torch
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import yaml
from aimlsse_api.client import GroundDataClient, SatelliteDataClient
from aimlsse_api.data import *
from aimlsse_api.data import Credentials, QueryStates
from aimlsse_api.data.metar import *
from bs4 import BeautifulSoup
from dacite import from_dict
from dateutil.relativedelta import relativedelta
from matplotlib import colormaps
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.ticker import FormatStrFormatter
from metar import Metar
from ml_commons import *
from numpy.typing import NDArray
from ptflops import get_model_complexity_info
from rasterio import CRS, plot
from rasterio.crs import CRS
from rasterio.enums import Resampling
from requests.auth import HTTPBasicAuth
from sentinelsat import SentinelAPI
from sentinelsat.sentinel import SentinelAPI
from shapely import (Geometry, GeometryCollection, MultiPolygon, Point,
                     Polygon, box)
from shapely.ops import unary_union
from sklearn.cluster import DBSCAN
from sklearn.metrics import (ConfusionMatrixDisplay, confusion_matrix,
                             precision_recall_fscore_support)
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision.models import (EfficientNet, EfficientNet_V2_S_Weights,
                                Swin_V2_S_Weights, SwinTransformer,
                                efficientnet_v2_s, swin_v2_s)
from tqdm import tqdm
from tqdm.auto import tqdm
