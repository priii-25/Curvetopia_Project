from data_preprocessing.preprocess import read_csv, plot, polylines2svg

csv_path = 'src\curve_completion\occlusion1_sol.csv'
svg_path = 'src/data_preprocessing/isolated.svg'

path_XYs = read_csv(csv_path)
#plot(path_XYs)
polylines2svg(csv_path,svg_path)