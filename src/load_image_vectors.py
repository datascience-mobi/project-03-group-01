import gzip
from typing import List


def load_gz(path) -> list:
    """
    Loads bytes from input .csv.gz file (single line)
    :param path: path to input file (.gz)
    :return: list of CsvImage Objects
    """
    input_file = gzip.open(path, 'rb')

    try:
        data = input_file.read()
    finally:
        input_file.close()

    # Translate data (byte) to data (str)
    data = data.decode("utf-8")

    # Create list, where each element is one image
    data_vector = data.split("\n")

    # Make sure not to return empty or corrupted lines - optional
    # count=0
    # for dat in data_vector:
    #     if not (len(dat.split(",")))==785:
    #         print("FAIL"+str(count))
    #     count+=1

    # Remove empty last element, was created since the input string ends with "\n"
    data_vector.pop(len(data_vector) - 1)

    # directly return loaded image as CsvImage object list
    return get_image_object_list(data_vector)


def load_csv(path) -> list:
    """
    Loads strings from input .csv (multiple lines)
    :param path: path to .csv
    :return: list of CsvImage Objects
    """
    data_list = list()  # type: List[str]

    # Add each line from input .csv to data_list
    with open(path) as infile:
        for line in infile:
            line = line.replace("\n", "")
            data_list.append(line)

    # directly return as CsvImage object list
    return get_image_object_list(data_list)


def get_image_object_list(data_list) -> list:
    """
    Makes list of CsvImage objects out of a list of image strings
    :param data_list: list of image strings
    :return: list of CsvImage objects
    """
    # create a list to store CsvImage objects in
    image_list = list()

    # for every image string: create CsvImage object and append to image_list
    for data in data_list:
        image = CsvImage(data)
        image_list.append(image)

    # to manually check if list is as long as expected
    print(len(image_list))
    return image_list


def get_pixel_list(strings) -> list:
    values = strings.split(",")
    values.pop(0)
    image = list()
    for pixel in values:
        image.append(int(pixel))
    return image


class CsvImage:
    """
    Contains the images label (which digit it represents) and its list of intensity values
    """

    def __init__(self, input_image):
        """
        Initializes CsvImage object, chops of first value (value) as int, saves rest into pixel list
        :param input_image: string of label and intensities, separated by ","
        """
        # Split input string ("0,0,5 ... 0,0") after each ",", returns a list of strings
        values = input_image.split(",")

        # label (accessible via object_name.label) is set to the first number of the input vector
        self.label = int(values[0])

        # first value is removed from the list because it is now stored as label
        # -> values list only contains intensity values now
        values.pop(0)

        # image (accessible via object_name.image) is a list, containing all values components as integer
        self.image = list()
        for pixel in values:
            self.image.append(int(pixel))
