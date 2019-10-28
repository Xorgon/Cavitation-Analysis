def array_to_csv(path, filename, array):
    file = open(path + filename, "w")
    for entry in array:
        line = ""
        for part in entry:
            line += str(part) + ","
        line += "\n"
        file.write(line)
    file.close()
