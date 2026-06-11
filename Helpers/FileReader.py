def file_reader(file_name):

    while True:

        row = file_name.readline()

        if not row:

            break

        yield row

