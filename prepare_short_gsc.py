import os
import os
import random
import shutil

gsc_dataset_path = "/home/sarthak/Projects/Augnito/datasets/google-speech-commands-v2/"
short_gsc_path = "/home/sarthak/Projects/Augnito/datasets/short_gsc/"
word_list = [x for x in os.listdir(gsc_dataset_path) if os.path.isdir(gsc_dataset_path + x)]
for word in word_list:
    source = gsc_dataset_path + word
    dest = short_gsc_path + word
    files = os.listdir(source)
    no_of_files = len(files) // 5
    if(not os.path.isdir(dest)):
        os.mkdir(dest)
    print("Copying %i files out of %i files for %s" %(no_of_files, len(files), word))
    for file_name in random.sample(files, no_of_files):
        shutil.copy(os.path.join(source, file_name), dest)

    onlyfiles = [f for f in os.listdir(dest) if os.path.isfile(os.path.join(dest, f))]
    val = int(round(len(onlyfiles)*0.1))
    test = int(round(len(onlyfiles)*0.1))

    print("Total files %i" % (len(onlyfiles)))

    random.shuffle(onlyfiles)
    print(val)
    val_lst = onlyfiles[0:val]
    test_lst = onlyfiles[val+1:test+val+1]

    print("Validation %i Test %i" %(len(val_lst), len(test_lst)))

    testfile = open(short_gsc_path+"testing_list.txt", "a")
    for element in test_lst:
        testfile.write(word + '/' + element + "\n")
    testfile.close()

    valfile = open(short_gsc_path+"validation_list.txt", "a")
    for element in val_lst:
        valfile.write(word + '/' + element + "\n")
    valfile.close()

