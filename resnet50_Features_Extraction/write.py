import numpy as np
import csv

def writeARFF(file_name, np_array_list, output_test):
    output_filename = file_name + '.arff'
    with open(output_filename, "w") as fp:
        fp.write('@RELATION ' + file_name)

        for i in range(np_array_list[0].size):
            fp.write("\n@ATTRIBUTE x%d REAL" % (i+1))

        fp.write('\n@ATTRIBUTE class {Covid19,Healthy}')

        fp.write('\n\n@DATA\n')

        for i in range(len(np_array_list)):
            fp.write('\n')
            for j in range(np_array_list[i].size):
                fp.write(str(np_array_list[i][j]) + ',')

            fp.write(output_test[i])
    fp.close()