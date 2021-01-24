import axios from 'axios';
import * as turf from '@turf/turf';

// const topojson = require('topojson');
const d3 = Object.assign(
  {},
  require('d3'),
  require('d3-geo'),
  require('d3-geo-projection'),
  require('d3-geo-voronoi'),
  require('d3-scale'),
  require('d3-scale-chromatic'),
);

const projection = d3.geoOrthographic();
// const projection = d3.geoBromley();
const path = d3.geoPath().projection(projection);
const graticule = d3.geoGraticule();

const svg = d3.select('svg');

function addControls() {
  const angles = ['λ', 'φ', 'γ'];

  angles.forEach(function (angle, index) {
    d3.select('#rotation')
      .append('div')
      .attr('class', 'angle-label angle-label-' + index)
      .html(angle + ': <span>0</span>');

    d3.select('#rotation')
      .append('input')
      .attr('type', 'range')
      .attr('class', 'angle angle-' + index)
      .attr('min', '-180')
      .attr('max', '180')
      .attr('step', '1')
      .attr('value', '0');
  });

  d3.selectAll('input').on('input', function () {
    // get all values
    const nums = [];

    d3.selectAll('input').each(function () {
      nums.push(+d3.select(this).property('value'));
    });
    update(nums);

    svg.selectAll('path').attr('d', path);
  });

  function update(eulerAngles) {
    angles.forEach(function (angle, index) {
      d3.select('.angle-label-' + index + ' span').html(
        Math.round(eulerAngles[index]),
      );
      d3.select('.angle-' + index).property('value', eulerAngles[index]);
    });

    projection.rotate(eulerAngles);
  }
}

function buildGlobe() {
  axios.get('http://localhost:8080/data/world_low.json').then((res) => {
    const world = res.data;

    svg
      .selectAll('.country')
      .data(world.features)
      .enter()
      .insert('path', '.graticule')
      .attr('class', 'country')
      .attr('d', path);
  });
}

function buildDelauny(data) {
  const v = d3.geoVoronoi()(data);

  const cityCoords = data.features.map(
    (feature) => feature.geometry.coordinates,
  );

  const geoDelaunay = d3.geoDelaunay(cityCoords);

  const polygons = geoDelaunay.triangles.map((tri) => {
    const coords = [
      data.features[tri[0]].geometry.coordinates,
      data.features[tri[1]].geometry.coordinates,
      data.features[tri[2]].geometry.coordinates,
      data.features[tri[0]].geometry.coordinates,
    ];

    return turf.polygon([coords]);
  });

  const centroids = polygons.map((feature) => {
    return turf.centroid(feature);
  });

  const geoCentroids = polygons.map((feature) => {
    const coords = d3.geoCentroid(feature);
    if (coords.includes[NaN]) {
      console.log('ohh', feature);
    }

    return turf.point(coords);
  });

  console.log(geoCentroids);

  svg
    .append('path')
    .attr('id', 'sphere')
    .datum({ type: 'Sphere' })
    .attr('d', path);

  svg
    .append('path')
    .datum(graticule)
    .attr('class', 'graticule')
    .attr('d', path);

  svg
    .append('path')
    .datum(graticule.outline)
    .attr('class', 'graticule outline')
    .attr('d', path);

  svg
    .append('g')
    .attr('class', 'triangles')
    .selectAll('path')
    .data(v.triangles().features)
    .enter()
    .append('path')
    .attr('d', path)
    .attr('opacity', 0.2)
    .attr('fill', function () {
      return d3.hsl(Math.random() * 360, 0.8, 0.65);
    });

  svg
    .append('g')
    .attr('class', 'sites')
    .selectAll('path')
    .data(data.features)
    .enter()
    .append('path')
    .attr('d', path);

  svg
    .append('g')
    .attr('class', 'geoCentroid')
    .selectAll('path')
    .data(geoCentroids)
    .enter()
    .append('path')
    .attr('d', path);

  svg
    .append('g')
    .attr('class', 'centroid')
    .selectAll('path')
    .data(centroids)
    .enter()
    .append('path')
    .attr('d', path);

  // gentle animation
  // d3.interval(function (elapsed) {
  //   projection.rotate([elapsed / 150, 0]);
  //   svg.selectAll('path').attr('d', path);
  // }, 50);
}

axios
  .get('http://localhost:8080/data/cities_pop_10000000.geojson')
  .then((res) => {
    // handle success
    buildGlobe();
    buildDelauny(res.data);
    addControls();
  })
  .catch((error) => {
    // handle error
    console.log(error);
  })
  .then(() => {
    // always executed
  });
