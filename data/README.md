# Data
The given documentation explains the processing of the data. The two dataset CommonRoad and OpenDD are currently implemented. The research is evaluated on the CommmonRoad dataset.

## CommonRoad 
The dataset used in this research is open-source available to download, see [README](../README.md).

### Manual Dataset creation
1. Clone the repo https://gitlab.lrz.de/tum-cps/commonroad-scenarios.
2. Run the script `extract_CR_from_repo` in [processing.py](../utils/processing.py) to extract the .xml-files from the repo. Pass the path to the downloaded repo as argument to the script.
3. Next, you can create your own dataset with the script [Dataset_CR.py](utils/Dataset_CR.py) in `utils`. Note to pass the correct path to the location of the data (argument `processed_file_folder`) you extracted in the step before. The `processed_file_folder` has to contain either the folder `raw` with `commonroad-scenarios` as .xml or the already processed .pt-files in the folder `processed`.

## OpenDD
1. Download the data from https://l3pilot.eu/data/opendd/ into the folder raw
2. Arrange the folder structure as follows: 
		> data/raw/
			>rdb1
				>geo-referenced_images_rdb1
					>rdb1.pgw
					>rdb1.png
					>rdb1_drivable_area.png
				>trajectories_rdb1_v3.sqlite
				>split_definition
				.
				.
				.
			>rdb2
			.
			.
			.
			>rdb7
			>split_definition
3. Remember to copy the split_definition folder from any of the 7 roundabouts to folder data/raw/ (The split_definition folder is equivalent for every rdb)
4. Check whether the geo reference files of all roundabouts (raw/rdbx/geo-referenced_images_rdbx/rdb1.pgw) are really of type '.pgw'. When downloading the data from "https://l3pilot.eu/data/opendd", the file of rdb6 might have a different type. However, this can easily be fixed by simply renaming the file with '.pgw' at the end.
5. Run the script [Dataset_OpenDD.py](utils/Dataset_OpenDD.py) in `utils`.
