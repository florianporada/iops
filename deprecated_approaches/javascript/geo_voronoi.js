const fs = require('fs');
const d3 = Object.assign({}, require('d3'), require('d3-geo-voronoi'));

const dataset = process.argv.slice(2)[0];
const filename = dataset.match(/([^\/]+)(?=\.\w+$)/)[0];

const rawdata = fs.readFileSync(dataset || './cities_pop_10000000.geojson');
const fc = JSON.parse(rawdata);

v = d3.geoVoronoi()(fc);

fs.writeFileSync(`geo_delaunay_${filename}.geojson`, JSON.stringify(v.triangles()));

console.log(`geo_delaunay_${filename}.geojson`);
