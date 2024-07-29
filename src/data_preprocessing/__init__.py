from data_preprocessing.preprocess import read_csv, polylines2svg, plot

csv_path = 'src/data_preprocessing/isolated.csv'
svg_path = 'path/to/your/output.svg'

path_XYs = read_csv(csv_path)
plot(path_XYs)
# polylines2svg(path_XYs, svg_path)
