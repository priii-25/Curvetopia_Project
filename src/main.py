from data_preprocessing import read_csv, convert_to_polylines

data = read_csv("data/raw/example.csv")
polylines = convert_to_polylines(data)
