import geetools
import ee
import sys

"""
In /data/bw462/satellite/data_usa verified that all of the cdl files in a span
(at least Jun-Aug) are the same, so just going to save 1 of them
"""

ee.Initialize()
YEARS         = [2017, 2014, 2015, 2016, 2018, 2019, 2020];
MONTHS        = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
DAYS_IN_MONTH = [31,28,31,30,31,30,31,31,30,31,30,31];
IM_SIZE       = 26880; #  in meters, so this is equal to 16 224x224 pixel tiles
loc_names =['cali1', 'cali2', 'cali3', 'cali4', 'cali5', 'cali6', 'cali7', 'cali8',
              'cali9', 'cali10', 'cali11', 'cali12',
              'wash1', 'wash2', 'wash3', 'wash4', 'wash5', 'wash6', 'wash7', 'wash8',
              'wash9',
              'ny1', 'ny2', 'ny3', 'ny4', 'ny5', 'ny6',
              'corn1', 'corn2', 'corn3', 'corn4', 'corn5', 'corn6', 'corn7', 'corn8',
              'corn9', 'corn10', 'corn11', 'corn12',
              'south1', 'south2', 'south3', 'south4', 'south5', 'south6', 'south7', 'south8',
              'south9', 'south10'];
coords = [[-119.04438566867341, 35.24879376497997], [-119.34101652804841, 35.74075044289647], [-119.33220531967768, 36.23524987466366], [-119.94194653061518, 36.22638783534086], [-120.37041332749018, 36.71141332502223], [-119.76616528061518, 36.71141332502228], [-120.25505688217768, 37.19865899917554], [-120.84831860092768, 37.19865899917554], [-120.97940824196674, 37.68495351617897], [-121.47506071398614, 38.176999669789254], [-121.76722434175836, 38.76936717601746], [-122.01014510619133, 39.54908720131019], [-120.37508933614559, 46.429264743149076], [-118.40853660177059, 46.28520072633207], [-119.71590964864559, 47.09903680573779], [-118.78207175802059, 47.03917312387078], [-117.27694480489559, 47.21110001057907], [-117.09017722677059, 46.723785404667034], [-116.35958640645809, 46.3041782185927], [-119.66097800802059, 46.17880523038064], [-118.77108542989559, 45.79713725967044], [-78.58071104520563, 43.130226979568825], [-77.91603819364313, 43.1342357686952], [-78.05611387723688, 42.643224786952125], [-76.34224668973688, 42.830833369504084], [-75.55397764676813, 42.97367927151397], [-75.82863584989313, 43.790103078898596], [-85.8672330241077, 41.05248082656611], [-86.7021939616077, 40.990314873921044], [-87.4712369303577, 40.79930686483558], [-88.1414029459827, 40.41563775934021], [-88.5698697428577, 40.965432064601565], [-88.9324185709827, 41.66680640796005], [-89.36402188594494, 40.10810615860227], [-90.31983243281994, 41.20819760132532], [-92.66441024981148, 43.452513033997036], [-94.45518173418648, 43.7786229571194], [-95.12534774981148, 44.919115273905625], [-95.93833603106148, 45.507312833146706], [-88.92191162717067, 36.56266604613745], [-89.05374756467067, 36.05804748437661], [-89.80081787717067, 36.67729651302157], [-89.91068115842067, 36.03139823617512], [-90.53690186154567, 35.40702791538179], [-89.33939209592067, 35.51441002764275], [-91.08621826779567, 34.84092603462239], [-90.40506592404567, 34.38884707258689], [-90.57601030804959, 33.83398821816854], [-91.46590288617459, 32.63944407720628]]
"""
Use below to start from loc that we terminated during previously
"""
completed_years = [2017, 2014]
completed_regions_for_partial_year = ['cali', 'wash', 'ny']

# tmp_start_index = loc_names.index('wash7')
# loc_names = loc_names[tmp_start_index:]
# coords = coords[tmp_start_index:]

def maskL8sr(image):
  ## Bits 3 and 5 are cloud shadow and cloud, respectively.
  # mask1 = geetools.cloud_mask.landsatSR(['cloud'])
  # mask2 = geetools.cloud_mask.landsatSR(['shadow'])
  cloudShadowBitMask = (1 << 3);
  cloudsBitMask = (1 << 5);
  ## Get the pixel QA band.
  qa = image.select('pixel_qa');
  ## Both flags should be set to zero, indicating clear conditions.
  # mask = cloudShadowBitMask;
  # mask = qa.bitwiseAnd(cloudShadowBitMask).eq(0).and(qa.bitwiseAnd(cloudsBitMask).eq(0));
  # mask = qa.bitwiseAnd(cloudShadowBitMask).eq(0) and  qa.bitwiseAnd(cloudsBitMask).eq(0)
  shadow_mask = qa.bitwiseAnd(cloudShadowBitMask).eq(0)
  cloud_mask = qa.bitwiseAnd(cloudsBitMask).eq(0)
  mask = shadow_mask.bitwiseAnd(cloud_mask)
  return image.updateMask(mask)

def main(point, folder, name, y, m):
  ## Which bands to select
  bands = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B10', 'B11']
  ## build string for date filter of imagecollection on line 37
  year = str(y);
  month = str(m) if m>=10 else '0' + str(m)
  end_day = str(DAYS_IN_MONTH[m-1])
  start = year+'-'+month+'-'+'01';
  end = year+'-'+month+'-'+end_day;
  ## Get Landsat reflectance data
  landsatCloudMaskedImage = ee.ImageCollection('LANDSAT/LC08/C01/T1_SR').filterDate(start, end).map(maskL8sr).select(bands).median().unmask();
  ## get corresponding ground truth
  gt_coll = ee.ImageCollection('USDA/NASS/CDL').filter(ee.Filter.date(year+'-01-01', year+'-12-31')).first();
  cropLandcover = gt_coll.select('cropland');
  ## create region to crop
  sample = point.buffer(IM_SIZE).bounds();
  ## display results
  visParams = {
    'bands': ['B4', 'B3', 'B2'],
    'min': 0,
    'max': 3000,
    'gamma': 1.4,
  };
  # Map.addLayer(landsatCloudMaskedImage.clip(sample), visParams, 'mosaic');
  ## Map.addLayer(cropLandcover.clip(sample), {}, 'ground truth')
  ## export results
  maxPixels = 6000000000;
  task = geetools.batch.Export.image.toDrive(
    image=landsatCloudMaskedImage,
    description='mosaic'+ '-' + name + '-'+year+'-'+month,
    folder=folder,
    scale=30,
    region=sample,
    crs='EPSG:4326',
    maxPixels=maxPixels)
  task.start()
  if m==7:#  Just need one per year so choosing smack in the middle of growth just to hedge
      cdl_task = geetools.batch.Export.image.toDrive(
        image=cropLandcover,
        description='ground_truth'+ '-' + name +'-'+year,
        folder=folder,
        scale=30,
        region=sample,
        crs='EPSG:4326',
        maxPixels=maxPixels)
      cdl_task.start()



next_year_partial = False
for year in YEARS:
    # Skip over fully completed years
    # Note that next year is partial
    if year in completed_years:
        next_year_partial = True
        continue
    for coord,name in zip(coords, loc_names):
        # If this year is partial then check for any completd regions in name of loc
        if next_year_partial:
            if any(region in name for region in completed_regions_for_partial_year):
                continue
        point = ee.Geometry.Point(coord)
        for month in MONTHS:
            print(name, year, month)
            main(point, 'earth_engine', name, year, month)
            sys.stdout.flush()
    # If we didn't hit above year continue then have all of next year
    next_year_partial = False
