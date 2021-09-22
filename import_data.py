import csv

def cleanup_data(data):
    #column1 seems useless, just "#NAME" remove it.
    #column15 is a string with value of A/N, so map to 0/1
    #column16 is a string with value of E/W, so map to 0/1
    #column21 is a string with value of A/N, so map to 0/1
    #otherwise, all other values must be a float.
    clean_data = []
    header = data[0]
    for row in data[1:]:
        cleaned_row = row
        for idx in [14,20]:
            if cleaned_row[idx]=="A":
                cleaned_row[idx]=0.0
            elif cleaned_row[idx]=="N":
                cleaned_row[idx]=1.0
            else:
                raise ValueError("Cant cleanup {}".format(cleaned_row[idx]))

        if cleaned_row[15]=="E":
            cleaned_row[15]=0.0
        elif cleaned_row[15]=="W":
            cleaned_row[15]=1.0
        else:
            raise ValueError("Cant cleanup {}".format(cleaned_row[idx]))

        #otherwise, force float.
        cleaned_row = cleaned_row[1:]
        isvalid = True
        for idx,val in enumerate(cleaned_row):
            if isinstance(val, str) and val.isnumeric():
                cleaned_row[idx] = float(val)
            elif isinstance(val,int) or isinstance(val,float):
                continue
            else:
                isvalid = False
                break

        if isvalid:
            clean_data.append(cleaned_row)

    return clean_data,header[1:]


def import_baseball_data(file):
    with open(file, newline='') as f:
        reader = csv.reader(f)
        data = list(reader)
        return cleanup_data(data)
