import sys
import json

PRE_COLS_COUNT = 1
FIRST_PRE_LEN = 15
LAST_SUF_LEN = 3

def unstack_multi_cols(path):
    file = open(path)
    to_file = open(path[:-4] + "_unstack.csv", 'w')
    count = 0
    write_count = 0
    for line in file:
        if count == 0:
            write_count = 0
        else:
            cols = line.strip('\r\n').split(',')
            new_line_pre = ','.join(cols[:1])
            line = ','.join(cols[1:])
            line = line[1:-1]
            line = line.replace("'", '"').replace(" ", "").replace("\t", " ")
            line = json.loads(line)

            key_list = line.keys()
            shop_list = line[key_list[0]].keys()

            if write_count == 0:
                to_file.write("row_id,shop_id,%s\n" % ','.join(key_list))
                write_count += 1

            for shop_id in shop_list:
                new_line = [new_line_pre, shop_id]
                for key in key_list:
                    new_line.append(str(line[key][shop_id]))
                to_file.write("%s\n" % ','.join(new_line))
                write_count += 1

        count += 1
        if count % 10000 == 0:
            print "read: %s rows,  write: %s rows" % (count, write_count)
    print "read: %s rows,  write: %s rows" % (count, write_count)


if __name__ == "__main__":
    unstack_multi_cols("../data/test.csv")
