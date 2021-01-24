import turf from '@turf/turf';
import fs from 'fs';

const extent = [-179.999, -89.99, 179.999, 89.99].map((el) =>
  el > 0 ? el - 50 : el + 50,
);

console.log(extent);
// const extent = [
//   13.091992716067702,
//   52.33488609760638,
//   13.742786470433,
//   52.67626223889507,
// ];
const cellSide = 1;
const options = { units: 'kilometers' };

const grid = turf.pointGrid(extent, cellSide, options);

console.log(`Points: ${grid.features.length}`);

fs.writeFileSync('./grid.geojson', JSON.stringify(grid));
