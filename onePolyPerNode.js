// 
// 
// Impor the provided shapefile with 80 fields as asset.
// The steps for how to do so is shown in the powerpoint file here:
// https://github.com/HNoorazar/KC/blob/main/remoteSensing.pptx
// 

// The following line converts the shapefile into a FeatureCollection
// that is useable by your script.
//
var myShapeFile = ee.FeatureCollection(Grant80Fields);
print("Number of fiels in myShapeFile is ", myShapeFile.size());

// Define the start and the end of the time period you want to 
// read. The WSDA crop classification in the shapefile is from 2017, 
// so, we must use 2017.
// Otherwise, your images and their labels will not be consistent.
//
var start_date = '2017-01-01';
var end_date = '2017-12-31';
    

////////////////////////////////////////////////////////////////////////////////////////
///
///                           functions definitions start
///
////////////////////////////////////////////////////////////////////////////////////////
///
///  Function to mask clouds using the Sentinel-2 QA band.
///

//
// Toss cloudly pixels Function.
//
function maskS2clouds(image) {
  // QA60 is a band storing the metdata related to clouds.
    var qa = image.select('QA60');

    // Bits 10 and 11 are clouds and cirrus, respectively.
    var cloudBitMask = 1 << 10;
    var cirrusBitMask = 1 << 11;

    // Both flags should be set to zero, indicating clear conditions.
    var mask = qa.bitwiseAnd(cloudBitMask).eq(0).and(
                        qa.bitwiseAnd(cirrusBitMask).eq(0));

    // Return the masked and scaled data, without the QA bands.
    return image.updateMask(mask).divide(10000)
                          .select("B.*")
                          .copyProperties(image, ["system:time_start"]);
}

////////////////////////////////////////////////
///
/// add Day of Year (1 to 365/366) to an image
/// 

function addDate_to_image(image){
  var doy = image.date().getRelative('day', 'year');
  var doyBand = ee.Image.constant(doy).uint16().rename('doy');
  doyBand = doyBand.updateMask(image.select('B8').mask());

  return image.addBands(doyBand);
}

////////////////////////
///
/// add Day of Year to an imageCollection/Time Series by using map(.) function
/// to recursively call the function addDate_to_image(.) defined above
//
//
function addDate_to_collection(collec){
  var C = collec.map(addDate_to_image);
  return C;
}

////////////////////////////////////////////////
///
///
/// add NDVI to an image

function addNDVI_to_image(image) {
  var ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI');
  return image.addBands(ndvi);
}

////////////////////////////////////////////////
///
///
/// add NDVI to an imageCollection by using map(.) function
/// to recursively call the function addDate_to_image(.) defined above
//
function add_NDVI_collection(image_IC){
  var NDVI_IC = image_IC.map(addNDVI_to_image);
  return NDVI_IC;
}

////////////////////////////////////////////////
///
///
/// Similar to NDVI functions defined above, we 
/// define and add EVI to an image

function addEVI_to_image(image) {
  var evi = image.expression(
                      '2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1.0))', {
                      'NIR': image.select('B8'),
                      'RED': image.select('B4'),
                      'BLUE': image.select('B2')
                  }).rename('EVI');
  return image.addBands(evi);
}

////////////////////////////////////////////////
///
///
/// add EVI to an imageCollection

function add_EVI_collection(image_IC){
  var EVI_IC = image_IC.map(addEVI_to_image);
  return EVI_IC;
}

////////////////////////////////////////////////
///
///


/// Extract ImageCollection/Time-Series for one field
// out of the 80 fields in the shapefile. And this function
// will be used recursively later to extract time series of
// all of the 80 fields.
//
function extract_sentinel_IC(a_feature){
  // a_feature: is one polygon and its charachteristics (Acres, Irrigation) in the shapefile 
  
    var geom = a_feature.geometry();
    var newDict = {'original_polygon_1': geom};
    
    var imageC = ee.ImageCollection('COPERNICUS/S2')
                .filterDate(start_date, end_date)
                .filterBounds(geom)
                // Clip(.) function  is clipping the boundary of the given field.
                .map(function(image){return image.clip(geom)})
                .filter('CLOUDY_PIXEL_PERCENTAGE < 90')
                .sort('system:time_start', true);
    
    // toss out cloudy pixels
    imageC = imageC.map(maskS2clouds);
    
    // pick up some bands
    imageC = imageC.select(['B8', 'B4', 'B3', 'B2']);
    
    // add DoY as a band
    imageC = addDate_to_collection(imageC);
    
    // add NDVI as a band
    imageC = add_NDVI_collection(imageC);
    
    // add EVI as a band
    imageC = add_EVI_collection(imageC);
    
    // add original geometry to each image
    // we do not need to do this really:
    imageC = imageC.map(function(im){return(im.set(newDict))});
    
    // add original geometry and WSDA data as a feature to the collection
    imageC = imageC.set({ 'original_polygon': geom,
                          'WSDA':a_feature
    
                        });

  return imageC;
}


//
// The function below (mosaic_and_reduce_IC_mean(.)) stiches
// the tiles that cover a polygon to create a complete image of a field
// and then reduces (i.e. takes averages of all) NDVI pixels values into one value for
// the field.
//

function mosaic_and_reduce_IC_mean(an_IC){
  an_IC = ee.ImageCollection(an_IC);
  
  var reduction_geometry = ee.Feature(ee.Geometry(an_IC.get('original_polygon')));
  var WSDA = an_IC.get('WSDA');
  var start_date_DateType = ee.Date(start_date);
  var end_date_DateType = ee.Date(end_date);
  //######**************************************
  // Difference in days between start and end_date

  var diff = end_date_DateType.difference(start_date_DateType, 'day');

  // Make a list of all dates
  var range = ee.List.sequence(0, diff.subtract(1)).map(function(day){
                                    return start_date_DateType.advance(day,'day')});

  // Funtion for iteraton over the range of dates
  function day_mosaics(date, newlist) {
    // Cast
    date = ee.Date(date);
    newlist = ee.List(newlist);

    // Filter an_IC between date and the next day
    var filtered = an_IC.filterDate(date, date.advance(1, 'day'));

    // Make the mosaic
    var image = ee.Image(filtered.mosaic());

    // Add the mosaic to a list only if the an_IC has images
    return ee.List(ee.Algorithms.If(filtered.size(), newlist.add(image), newlist));
  }

  // Iterate over the range to make a new list, and then cast the list to an imagecollection
  var newcol = ee.ImageCollection(ee.List(range.iterate(day_mosaics, ee.List([]))));
  //print("newcol 1", newcol);
  //######**************************************

  var reduced = newcol.map(function(image){
                            return image.reduceRegions({
                                                        collection:reduction_geometry,
                                                        reducer:ee.Reducer.mean(), 
                                                        scale: 10
                                                      });
                                          }
                        ).flatten();
                          
  reduced = reduced.set({ 'original_polygon': reduction_geometry,
                        'WSDA':WSDA
                      });
  WSDA = ee.Feature(WSDA);
  WSDA = WSDA.toDictionary();
  
  // var newDict = {'WSDA':WSDA};
  reduced = reduced.map(function(im){return(im.set(WSDA))}); 
  return(reduced);
}

///
///                         functions definitions end
//
////////////////////////////////////////////////////////////////////////////////////////

var ImageCollection_2017 = myShapeFile.map(extract_sentinel_IC);

var all_fields_TS = ImageCollection_2017.map(mosaic_and_reduce_IC_mean);

// Export the file into YOUR google drive

// Export.table.toDrive({
//   collection: all_fields_TS.flatten(),
//   description:'KirtisClass_2017',
//   folder:"KirtisClass_2017",
//   fileFormat: 'CSV'
// });

// This is the Folder name and File name that will
// be used by Export.table.toDrive(.) to export your data
// 
var yourFileName = "PutYourNameHere";
var FolderName = "PutAFildeNameHere";

Export.table.toDrive({
  collection: all_fields_TS.flatten(),
  description: yourFileName,
  folder: FolderName,
  fileNamePrefix: yourFileName,
  fileFormat: 'CSV',
  // Provide the name of variables of interst to export.
  // Some of these names come from shapefile that we have added to imagecollection
  //
  selectors:["ID", "CropTyp", "doy", "EVI", 'NDVI']
});
