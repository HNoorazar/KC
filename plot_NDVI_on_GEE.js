
var myShapeFile = ee.FeatureCollection(potato_SF);
print("Number of fiels in myShapeFile is ", myShapeFile.size());
// print (myShapeFile);


var start_date = '2017-01-01';
var end_date = '2017-12-31';



////////////////////////////////////////////////////////////////////
//
//      functions
//
//
var addNDVI = function(image) {
  var ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI');
  return image.addBands(ndvi);
};

var add_NDVI_IC = function(image_IC){
  var NDVI_IC = image_IC.map(addNDVI);
  return NDVI_IC;
};

////////////////////////////////////////////////////////////////////
//
//      parameters
//
//
var needed_bands = ['B8', 'B4'];

var cloud_level = 1;
var plot_scale = 10;
////////////////////////////////////////////////////////////////////
//
//    read ImageCollection
//



var last_double_IC = ee.ImageCollection('COPERNICUS/S2')
                     .filterDate( start_date, end_date )
                     .filterBounds(myShapeFile)
                     .filterMetadata('CLOUDY_PIXEL_PERCENTAGE', "less_than", cloud_level)
                     .select(needed_bands);


last_double_IC = add_NDVI_IC(last_double_IC);


var last_NDVI_series = ui.Chart.image.doySeriesByYear({
              imageCollection: last_double_IC, 
              bandName: 'NDVI', 
              region: myShapeFile, 
              regionReducer: ee.Reducer.mean(),
              scale: plot_scale
              });
              

print(last_NDVI_series);

