cat ../dataset_x.dat d1/dataset_x.dat d2/dataset_x.dat d3/dataset_x.dat d4/dataset_x.dat d5/dataset_x.dat d6/dataset_x.dat d7/dataset_x.dat d8/dataset_x.dat d9/dataset_x.dat d10/dataset_x.dat > dataset_x.dat
cat ../dataset_y.dat d1/dataset_y.dat d2/dataset_y.dat d3/dataset_y.dat d4/dataset_y.dat d5/dataset_y.dat d6/dataset_y.dat d7/dataset_y.dat d8/dataset_y.dat d9/dataset_y.dat d10/dataset_y.dat > dataset_y.dat
cp dataset_x.dat ../dataset_x.dat
cp dataset_y.dat ../dataset_y.dat
rm dataset_x.dat
rm dataset_y.dat