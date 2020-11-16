
var SF = ee.FeatureCollection(SF_table);
print("Number of fiels in 2017 shapefile is ", SF_table.size());

////////////////////////////////////////////////////////////////////////////////////////
///
///                           functions definitions start
///
////////////////////////////////////////////////////////////////////////////////////////
///
///  Function to mask clouds using the Sentinel-2 QA band.
///

function maskS2clouds(image) {
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
///     add Year
///

function addYear_to_image(image){
  var year = image.date().get('year');
  var yearBand = ee.Image.constant(year).uint16().rename('image_year');
  return image.addBands(yearBand);
}

function addYear_to_collection(collec){
  var C = collec.map(addYear_to_image);
  return C;
}

////////////////////////////////////////////////
///
/// add Day of Year
///

function addDoY_to_image(image){
  var doy = image.date().getRelative('day', 'year');
  // var doy = ee.Date(image.get('system:time_start')).getRelative('day', 'year');
  var doyBand = ee.Image.constant(doy).uint16().rename('doy');
  doyBand = doyBand.updateMask(image.select('B8').mask());
  return image.addBands(doyBand);
}

function addDoY_to_collection(collec){
  var C = collec.map(addDoY_to_image);
  return C;
}

////////////////////////////////////////////////
///
/// add Date of image to an imageCollection
///

function add_Date_Time_image(image) {
  image = image.addBands(image.metadata('system:time_start').rename("system_start_time"));
  return image;
}

function add_Date_Time_collection(colss){
 var c = colss.map(add_Date_Time_image);
 return c;
}

////////////////////////////////////////////////
///
///         NDVI
///

function addNDVI_to_image(image) {
  var ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI');
  return image.addBands(ndvi);
}

function add_NDVI_collection(image_IC){
  var NDVI_IC = image_IC.map(addNDVI_to_image);
  return NDVI_IC;
}

////////////////////////////////////////////////
///
///         EVI
///

function addEVI_to_image(image) {
  var evi = image.expression(
                      '2.5 * ((NIR - RED) / (NIR + (6 * RED) - (7.5 * BLUE) + 1.0))', {
                      'NIR': image.select('B8'),
                      'RED': image.select('B4'),
                      'BLUE': image.select('B2')
                  }).rename('EVI');
  return image.addBands(evi);
}

function add_EVI_collection(image_IC){
  var EVI_IC = image_IC.map(addEVI_to_image);
  return EVI_IC;
}

////////////////////////////////////////////////
///
///       LSWI
///

function add_LSWI_to_image(image) {
  var LSWI = image.normalizedDifference(['B8A', 'B11']).rename('LSWI');
  return image.addBands(LSWI);
}

function add_LSWI_collection(image_IC){
  var LSWI_IC = image_IC.map(add_LSWI_to_image);
  return LSWI_IC;
}

////////////////////////////////////////////////
///
///          PSRI
///

function add_PSRI_to_image(image) {
  var psri = image.select('B4').subtract(image.select('B3')).divide(image.select('B8')).rename('PSRI');
  return image.addBands(psri);
}

function add_PSRI_collection(image_IC){
  var PSRI_IC = image_IC.map(add_PSRI_to_image);
  return PSRI_IC;
}

////////////////////////////////////////////////
///
///          BSI
///

function add_BSI_to_image(image) {
  var numerator = (image.select('B11').add(image.select('B4'))).subtract(image.select('B8').add(image.select('B2')));
  var denominator = (image.select('B11').add(image.select('B4'))).add(image.select('B8').add(image.select('B2')));
  var bsi = numerator.divide(denominator).rename('BSI');
  return image.addBands(bsi);
}

function add_BSI_collection(image_IC){
  var BSI_IC = image_IC.map(add_BSI_to_image);
  return BSI_IC;
}

////////////////////////////////////////////////
///
///         NDWI
///

function addNDWI_to_image(image) {
  var ndwi = image.normalizedDifference(['B3', 'B8']).rename('NDWI');
  return image.addBands(ndwi);
}

function add_NDWI_collection(image_IC){
  var NDWI_IC = image_IC.map(addNDWI_to_image);
  return NDWI_IC;
}

////////////////////////////////////////////////
///
///         Do the Job function
///

function extract_sentinel_IC(a_feature,start_date,end_date){
    var geom = a_feature.geometry(); //a_feature is a feature collection
    var newDict = {'original_polygon_1': geom};
    var imageC = ee.ImageCollection('COPERNICUS/S2')
                .filterDate(start_date, end_date)
                .filterBounds(geom)
                //.filterMetadata('CLOUDY_PIXEL_PERCENTAGE', "less_than", cloud_perc)
                .map(function(image){return image.clip(geom)})
                .filter(ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE', cloud_perc))
                //.filter('CLOUDY_PIXEL_PERCENTAGE < 70')
                .sort('system:time_start', true);
    
    // toss out cloudy pixels
    imageC = imageC.map(maskS2clouds);
    
    // pick up some bands
    // imageC = imageC.select(['B8', 'B4', 'B3', 'B2']);
    
    // add year as a band
    imageC = addYear_to_collection(imageC);
    
    // add DoY as a band
    imageC = addDoY_to_collection(imageC);
    
    imageC = add_Date_Time_collection(imageC);
    
    // add NDVI as a band
    imageC = add_NDVI_collection(imageC);
    
    // add EVI as a band
    imageC = add_EVI_collection(imageC);
    
    imageC = add_LSWI_collection(imageC);
    imageC = add_PSRI_collection(imageC);
    imageC = add_BSI_collection(imageC);
    imageC = add_NDWI_collection(imageC);

    // add original geometry to each image
    // we do not need to do this really:
    imageC = imageC.map(function(im){return(im.set(newDict))});
    
    // add original geometry and WSDA data as a feature to the collection
    imageC = imageC.set({ 'original_polygon': geom,
                          'WSDA':a_feature
                        });
    // imageC = imageC.sort('system:time_start', true);
    //imageC = imageC.map(add_NDVI_collection)
  return imageC;
}

function mosaic_and_reduce_IC_mean(an_IC,a_feature,start_date,end_date){
  an_IC = ee.ImageCollection(an_IC);
  //print('mosaic_start_date:',start_date);
  //var reduction_geometry = ee.Feature(ee.Geometry(an_IC.get('original_polygon')));
  var reduction_geometry = a_feature;
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
                                                        scale: 10//,
                                                        //tileScale: 16
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

// remove geometry on each feature before printing or exporting

var myproperties=function(feature){
  feature=ee.Feature(feature).setGeometry(null);
  return feature;
};


var xmin=-125.0;
var ymin = 45.0;
var xmax=-116.0;
var ymax = 49.0;
//var xmin=-120.0;
//var ymin = 47.0;
//var xmax=-119.9;
//var ymax = 47.1;
var xmed1 = (xmin + xmax) / 2.0;
var xmed2 = (xmin + xmax) / 2.0;

var WA1 = ee.Geometry.Polygon([[xmin, ymin], [xmin, ymax], [xmed1, ymax], [xmed1, ymin], [xmin, ymin]]);
var WA2 = ee.Geometry.Polygon([[xmed2, ymin], [xmed2, ymax], [xmax, ymax], [xmax, ymin], [xmed2, ymin]]);
var WA = [WA1,WA2];

var SF_regions = ee.FeatureCollection(WA);
var reduction_geometry = ee.FeatureCollection(SF);

print ("Number of fields in the shapefile is", SF_regions.size());


var wstart_date = '2016-08-01';
// var wend_date = '2019-03-31';
var wend_date = '2017-12-31';
var cloud_perc = 10;

var imageC = extract_sentinel_IC(SF_regions, wstart_date, wend_date);
var reduced = mosaic_and_reduce_IC_mean(imageC, reduction_geometry, wstart_date, wend_date);  
var featureCollection = reduced;

// var featureCollection = featureCollection.map(myproperties);

var outfile_name = 'Eastern_WA_2017_' + cloud_perc + "cloud_selectors";
Export.table.toDrive({
  collection: featureCollection,
  description:outfile_name,
  folder:"Eastern_WA_" + cloud_perc + "cloud",
  fileNamePrefix: outfile_name,
  fileFormat: 'CSV',
  selectors:["ID", "Acres", "BSI", "county", 
             "CropGrp", "CropTyp", "DataSrc", "doy", "EVI",
             "ExctAcr", "IntlSrD", "Irrigtn", 
             "LstSrvD", "LSWI", 'NDVI', "NDWI", "Notes", 
             "PSRI", "RtCrpTy", "Shap_Ar", "Shp_Lng", 
             "system_start_time", "TRS", "image_year", "B8"]
});


