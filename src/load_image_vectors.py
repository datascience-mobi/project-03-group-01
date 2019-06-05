import gzip
from typing import List


def load_gz(path) -> list:
    """
    Loads bytes from input .csv.gz file (single line)
    :param path: path to input file (.gz)
    :return: list of images (label + intensity values) as string, separated by ","
    """
    input_file = gzip.open(path, 'rb')
    # data = None # TODO: try does not create a new scope?
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
    print(len(data_vector))

    # directly return loaded image as CsvImage object list
    return get_image_object_list(data_vector)


def load_csv(path) -> list:
    """
    Loads strings from input .csv (multiple lines)
    :param path: path to .csv
    :return:
    """
    data_list = list()  # type: List[str]

    # Add each line from input .csv to data_list
    with open(path) as infile:
        for line in infile:
            line = line.replace("\n", "")
            data_list.append(line)

    print(len(data_list))  # Check if line count is as expected
    # print(data_list[0]) # Check if "\n" successfully removed

    # directly return as CsvImage object list
    return get_image_object_list(data_list)


def get_image_object_list(data_list):
    """
    Makes list of CsvImage objects out of a list of image strings
    :param data_list: list of image strings
    :return: list of CsvImage objects
    """
    image_list = list()
    simple_image_list = list()
    for data in data_list:
        image = CsvImage(data)
        image_list.append(image)
        simple_image_list.append(get_pixel_list(data))
    print(len(image_list))
    return image_list, simple_image_list


def get_pixel_list(strings) -> list:
    values = strings.split(",")
    #self.label = int(values[0])
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
        values = input_image.split(",")
        self.label = int(values[0])
        values.pop(0)
        self.image = list()
        for pixel in values:
            self.image.append(int(pixel))
