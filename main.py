import glob
import csv


if __name__ == '__main__':
    database = {}
    for name in glob.glob('cartoonset10k/*.csv'):
        with open(name, 'r') as file:
            image = {}
            for row in csv.reader(file):
                image[row[0]] = int(row[1].strip())
            database[name] = image
    print(database['cartoonset10k/cs1048486361028912.csv'])
