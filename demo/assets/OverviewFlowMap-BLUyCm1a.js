import{a as e,n as t,t as n}from"./jsx-runtime-D3jfb0Ew.js";import{a as r,c as i,d as a,i as o,l as s,o as c,r as l,s as u,u as d}from"./index-D9N1IK3d.js";import{a as f,c as p,d as m,f as h,l as g,m as _,o as v,r as y,s as b,u as x}from"./tile-layer-DufYXfbE.js";import{n as S,r as C}from"./overview-map-scene-DTcz8JhO.js";var w=`struct ArcUniforms {
  greatCircle: f32,
  useShortestPath: f32,
  numSegments: f32,
  widthScale: f32,
  widthMinPixels: f32,
  widthMaxPixels: f32,
  widthUnits: i32,
};

@group(0) @binding(auto) var<uniform> arc: ArcUniforms;
`,T=`layout(std140) uniform arcUniforms {
  bool greatCircle;
  bool useShortestPath;
  float numSegments;
  float widthScale;
  float widthMinPixels;
  float widthMaxPixels;
  highp int widthUnits;
} arc;
`,E={name:`arc`,source:w,vs:T,fs:T,uniformTypes:{greatCircle:`f32`,useShortestPath:`f32`,numSegments:`f32`,widthScale:`f32`,widthMinPixels:`f32`,widthMaxPixels:`f32`,widthUnits:`i32`}},D=`const ZERO_OFFSET: vec3<f32> = vec3<f32>(0.0, 0.0, 0.0);

struct Attributes {
  @location(0) instanceSourcePositions: vec3<f32>,
  @location(1) instanceSourcePositions64Low: vec3<f32>,
  @location(2) instanceTargetPositions: vec3<f32>,
  @location(3) instanceTargetPositions64Low: vec3<f32>,
  @location(4) instanceSourceColors: vec4<f32>,
  @location(5) instanceTargetColors: vec4<f32>,
  @location(6) instanceWidths: f32,
  @location(7) instanceHeights: f32,
  @location(8) instanceTilts: f32,
};

struct Varyings {
  @builtin(position) position: vec4<f32>,
  @location(0) color: vec4<f32>,
  @location(1) uv: vec2<f32>,
  @location(2) pickingColor: vec3<f32>,
  @location(3) isValid: f32,
};

fn paraboloid(
  distance: f32,
  sourceZ: f32,
  targetZ: f32,
  ratio: f32,
  height: f32
) -> f32 {
  let deltaZ = targetZ - sourceZ;
  let dh = distance * height;
  if (dh == 0.0) {
    return sourceZ + deltaZ * ratio;
  }
  let unitZ = deltaZ / dh;
  let p2 = unitZ * unitZ + 1.0;
  let dir = select(0.0, 1.0, deltaZ <= 0.0);
  let z0 = mix(sourceZ, targetZ, dir);
  let r = mix(ratio, 1.0 - ratio, dir);
  return sqrt(max(r * (p2 - r), 0.0)) * dh + z0;
}

fn getExtrusionOffset(lineClipspace: vec2<f32>, side: f32, width: f32) -> vec2<f32> {
  var direction = normalize(lineClipspace * project.viewportSize);
  direction = vec2<f32>(-direction.y, direction.x);
  return direction * side * width / 2.0;
}

fn getSegmentRatio(index: f32) -> f32 {
  return smoothstep(0.0, 1.0, index / max(arc.numSegments - 1.0, 1.0));
}

fn interpolateFlat(
  source: vec3<f32>,
  targetPosition: vec3<f32>,
  ratio: f32,
  height: f32,
  tiltDegrees: f32
) -> vec3<f32> {
  let distance = length(source.xy - targetPosition.xy);
  let z = paraboloid(distance, source.z, targetPosition.z, ratio, height);
  let tiltAngle = radians(tiltDegrees);
  let tiltDirection = normalize(targetPosition.xy - source.xy);
  let tilt = vec2<f32>(-tiltDirection.y, tiltDirection.x) * z * sin(tiltAngle);
  return vec3<f32>(mix(source.xy, targetPosition.xy, ratio) + tilt, z * cos(tiltAngle));
}

// Great circle interpolation
// http://www.movable-type.co.uk/scripts/latlong.html
fn getAngularDistance(source: vec2<f32>, targetPosition: vec2<f32>) -> f32 {
  let sourceRadians = radians(source);
  let targetRadians = radians(targetPosition);
  let sinHalfDelta = sin((sourceRadians - targetRadians) / 2.0);
  let sinHalfDeltaSquared = sinHalfDelta * sinHalfDelta;
  let a = sinHalfDeltaSquared.y +
    cos(sourceRadians.y) * cos(targetRadians.y) * sinHalfDeltaSquared.x;
  return 2.0 * asin(sqrt(a));
}

fn interpolateGreatCircle(
  source: vec3<f32>,
  targetPosition: vec3<f32>,
  source3D: vec3<f32>,
  target3D: vec3<f32>,
  angularDistance: f32,
  ratio: f32,
  height: f32
) -> vec3<f32> {
  var longitudeLatitude: vec2<f32>;

  // If the angular distance is PI, use linear interpolation. Otherwise use spherical interpolation.
  if (abs(angularDistance - PI) < 0.001) {
    longitudeLatitude = (1.0 - ratio) * source.xy + ratio * targetPosition.xy;
  } else {
    let a = sin((1.0 - ratio) * angularDistance);
    let b = sin(ratio * angularDistance);
    let p = source3D.yxz * a + target3D.yxz * b;
    longitudeLatitude = degrees(vec2<f32>(
      atan2(p.y, -p.x),
      atan2(p.z, length(p.xy))
    ));
  }

  let z = paraboloid(
    angularDistance * EARTH_RADIUS,
    source.z,
    targetPosition.z,
    ratio,
    height
  );
  return vec3<f32>(longitudeLatitude, z);
}

@vertex
fn vertexMain(
  attributes: Attributes,
  @builtin(vertex_index) vertexIndex: u32,
  @builtin(instance_index) instanceIndex: u32
) -> Varyings {
  geometry.worldPosition = attributes.instanceSourcePositions;
  geometry.worldPositionAlt = attributes.instanceTargetPositions;

  let segmentIndex = f32(vertexIndex / 2u);
  let segmentSide = select(-1.0, 1.0, vertexIndex % 2u == 1u);
  var segmentRatio = getSegmentRatio(segmentIndex);
  let previousRatio = getSegmentRatio(max(0.0, segmentIndex - 1.0));
  var nextRatio = getSegmentRatio(min(arc.numSegments - 1.0, segmentIndex + 1.0));
  // If this is the first point, use next - current as direction.
  var indexDirection = select(-1.0, 1.0, segmentIndex <= 0.0);
  var isValid = 1.0;

  var currentClip: vec4<f32>;
  var nextClip: vec4<f32>;

  if (
    (arc.greatCircle != 0.0 || project.projectionMode == PROJECTION_MODE_GLOBE) &&
    project.coordinateSystem == COORDINATE_SYSTEM_LNGLAT
  ) {
    let source = project_globe_(vec3<f32>(attributes.instanceSourcePositions.xy, 0.0));
    let targetPosition = project_globe_(vec3<f32>(attributes.instanceTargetPositions.xy, 0.0));
    let angularDistance = getAngularDistance(
      attributes.instanceSourcePositions.xy,
      attributes.instanceTargetPositions.xy
    );

    let previousPosition = interpolateGreatCircle(
      attributes.instanceSourcePositions,
      attributes.instanceTargetPositions,
      source,
      targetPosition,
      angularDistance,
      previousRatio,
      attributes.instanceHeights
    );
    var currentPosition = interpolateGreatCircle(
      attributes.instanceSourcePositions,
      attributes.instanceTargetPositions,
      source,
      targetPosition,
      angularDistance,
      segmentRatio,
      attributes.instanceHeights
    );
    var nextPosition = interpolateGreatCircle(
      attributes.instanceSourcePositions,
      attributes.instanceTargetPositions,
      source,
      targetPosition,
      angularDistance,
      nextRatio,
      attributes.instanceHeights
    );

    if (abs(currentPosition.x - previousPosition.x) > 180.0) {
      indexDirection = -1.0;
      isValid = 0.0;
    } else if (abs(currentPosition.x - nextPosition.x) > 180.0) {
      indexDirection = 1.0;
      isValid = 0.0;
    }
    nextPosition = select(nextPosition, previousPosition, indexDirection < 0.0);
    nextRatio = select(nextRatio, previousRatio, indexDirection < 0.0);

    if (isValid == 0.0) {
      // Split at the antimeridian.
      nextPosition.x += select(360.0, -360.0, nextPosition.x > 0.0);
      let ratio = (
        select(-180.0, 180.0, currentPosition.x > 0.0) - currentPosition.x
      ) / (nextPosition.x - currentPosition.x);
      currentPosition = mix(currentPosition, nextPosition, ratio);
      segmentRatio = mix(segmentRatio, nextRatio, ratio);
    }

    let currentPosition64Low = mix(
      attributes.instanceSourcePositions64Low,
      attributes.instanceTargetPositions64Low,
      segmentRatio
    );
    let nextPosition64Low = mix(
      attributes.instanceSourcePositions64Low,
      attributes.instanceTargetPositions64Low,
      nextRatio
    );
    let currentProjection = project_position_to_clipspace_and_commonspace(
      currentPosition,
      currentPosition64Low,
      ZERO_OFFSET
    );
    currentClip = currentProjection.clipPosition;
    nextClip = project_position_to_clipspace(nextPosition, nextPosition64Low, ZERO_OFFSET);
    geometry.position = currentProjection.commonPosition;
  } else {
    var sourceWorld = attributes.instanceSourcePositions;
    var targetWorld = attributes.instanceTargetPositions;
    if (arc.useShortestPath != 0.0) {
      sourceWorld.x = ((sourceWorld.x + 180.0) % 360.0) - 180.0;
      targetWorld.x = ((targetWorld.x + 180.0) % 360.0) - 180.0;
      let deltaLongitude = targetWorld.x - sourceWorld.x;
      if (deltaLongitude > 180.0) {
        targetWorld.x -= 360.0;
      }
      if (deltaLongitude < -180.0) {
        sourceWorld.x -= 360.0;
      }
    }

    let source = project_position_vec3_f64(
      sourceWorld,
      attributes.instanceSourcePositions64Low
    );
    let targetPosition = project_position_vec3_f64(
      targetWorld,
      attributes.instanceTargetPositions64Low
    );

    // Common x at longitude=-180.
    var antimeridianX = 0.0;
    if (arc.useShortestPath != 0.0) {
      if (project.projectionMode == PROJECTION_MODE_WEB_MERCATOR_AUTO_OFFSET) {
        antimeridianX = -(project.coordinateOrigin.x + 180.0) / 360.0 * TILE_SIZE;
      }
      let thresholdRatio = (antimeridianX - source.x) / (targetPosition.x - source.x);
      if (previousRatio <= thresholdRatio && nextRatio > thresholdRatio) {
        isValid = 0.0;
        indexDirection = sign(segmentRatio - thresholdRatio);
        segmentRatio = thresholdRatio;
      }
    }

    nextRatio = select(nextRatio, previousRatio, indexDirection < 0.0);
    var currentPosition = interpolateFlat(
      source,
      targetPosition,
      segmentRatio,
      attributes.instanceHeights,
      attributes.instanceTilts
    );
    var nextPosition = interpolateFlat(
      source,
      targetPosition,
      nextRatio,
      attributes.instanceHeights,
      attributes.instanceTilts
    );

    if (arc.useShortestPath != 0.0 && nextPosition.x < antimeridianX) {
      currentPosition.x += TILE_SIZE;
      nextPosition.x += TILE_SIZE;
    }

    currentClip = project_common_position_to_clipspace(vec4<f32>(currentPosition, 1.0));
    nextClip = project_common_position_to_clipspace(vec4<f32>(nextPosition, 1.0));
    geometry.position = vec4<f32>(currentPosition, 1.0);
  }

  geometry.uv = vec2<f32>(segmentRatio, segmentSide);
  geometry.pickingColor = picking_getPickingColorFromIndex(instanceIndex);

  let widthPixels = clamp(
    project_unit_size_to_pixel(attributes.instanceWidths * arc.widthScale, arc.widthUnits),
    arc.widthMinPixels,
    arc.widthMaxPixels
  );
#ifdef ANTIALIASING
  var offset = getExtrusionOffset(
#else
  let offset = getExtrusionOffset(
#endif
    (nextClip.xy - currentClip.xy) * indexDirection,
    segmentSide,
    widthPixels
  );
#ifdef ANTIALIASING
  let halfWidthPixels = length(offset);
  if (halfWidthPixels > 0.0) {
    // Keep the declared edge at abs(uv.y) == 1 while rasterizing the outer half of the centered
    // one-device-pixel coverage ramp.
    let coverageScale = 1.0 + 0.5 / project.devicePixelRatio / halfWidthPixels;
    offset *= coverageScale;
    geometry.uv.y *= coverageScale;
  }
#endif

  var output: Varyings;
  output.position = currentClip + vec4<f32>(project_pixel_size_to_clipspace(offset), 0.0, 0.0);
  let color = mix(attributes.instanceSourceColors, attributes.instanceTargetColors, segmentRatio);
  output.color = vec4<f32>(color.rgb, color.a * layer.opacity);
  output.uv = geometry.uv;
  output.pickingColor = geometry.pickingColor;
  output.isValid = isValid;
  return output;
}

@fragment
fn fragmentMain(varyings: Varyings) -> @location(0) vec4<f32> {
#ifdef ANTIALIASING
  let edgeCoord = abs(varyings.uv.y);
  let edgePixels = (1.0 - edgeCoord) / max(fwidth(edgeCoord), 1e-6);
#endif

  if (varyings.isValid == 0.0) {
    discard;
  }

#ifdef ANTIALIASING
  // Fragments outside the coverage ramp must not write depth or picking colors.
  if (edgePixels <= -SMOOTH_EDGE_RADIUS) {
    discard;
  }
#endif
  var color = varyings.color;
#ifdef ANTIALIASING
  // Feather one device pixel across the width. Arc segments meet lengthwise, so only soften the
  // two outer edges of the strip.
  color.a *= smoothedge(0.0, edgePixels);
#endif
  if (picking.isActive > 0.5) {
    if (!picking_isColorValid(varyings.pickingColor)) {
      discard;
    }
    return vec4<f32>(varyings.pickingColor, 1.0);
  }
  if (picking.isHighlightActive > 0.5) {
    let highlightedObjectColor = picking_normalizeColor(picking.highlightedObjectColor);
    if (picking_isColorZero(abs(varyings.pickingColor - highlightedObjectColor))) {
      let highlightAlpha = picking.highlightColor.a;
      let blendedAlpha = highlightAlpha + color.a * (1.0 - highlightAlpha);
      if (blendedAlpha > 0.0) {
        let highlightRatio = highlightAlpha / blendedAlpha;
        color = vec4<f32>(
          mix(color.rgb, picking.highlightColor.rgb, highlightRatio),
          blendedAlpha
        );
      }
    }
  }
  return deckgl_premultiplied_alpha(color);
}
`,O=`#version 300 es
#define SHADER_NAME arc-layer-vertex-shader
in vec4 instanceSourceColors;
in vec4 instanceTargetColors;
in vec3 instanceSourcePositions;
in vec3 instanceSourcePositions64Low;
in vec3 instanceTargetPositions;
in vec3 instanceTargetPositions64Low;
in float instanceWidths;
in float instanceHeights;
in float instanceTilts;
out vec4 vColor;
out vec2 uv;
out float isValid;
float paraboloid(float distance, float sourceZ, float targetZ, float ratio) {
float deltaZ = targetZ - sourceZ;
float dh = distance * instanceHeights;
if (dh == 0.0) {
return sourceZ + deltaZ * ratio;
}
float unitZ = deltaZ / dh;
float p2 = unitZ * unitZ + 1.0;
float dir = step(deltaZ, 0.0);
float z0 = mix(sourceZ, targetZ, dir);
float r = mix(ratio, 1.0 - ratio, dir);
return sqrt(r * (p2 - r)) * dh + z0;
}
vec2 getExtrusionOffset(vec2 line_clipspace, float offset_direction, float width) {
vec2 dir_screenspace = normalize(line_clipspace * project.viewportSize);
dir_screenspace = vec2(-dir_screenspace.y, dir_screenspace.x);
return dir_screenspace * offset_direction * width / 2.0;
}
float getSegmentRatio(float index) {
return smoothstep(0.0, 1.0, index / (arc.numSegments - 1.0));
}
vec3 interpolateFlat(vec3 source, vec3 target, float segmentRatio) {
float distance = length(source.xy - target.xy);
float z = paraboloid(distance, source.z, target.z, segmentRatio);
float tiltAngle = radians(instanceTilts);
vec2 tiltDirection = normalize(target.xy - source.xy);
vec2 tilt = vec2(-tiltDirection.y, tiltDirection.x) * z * sin(tiltAngle);
return vec3(
mix(source.xy, target.xy, segmentRatio) + tilt,
z * cos(tiltAngle)
);
}
float getAngularDist (vec2 source, vec2 target) {
vec2 sourceRadians = radians(source);
vec2 targetRadians = radians(target);
vec2 sin_half_delta = sin((sourceRadians - targetRadians) / 2.0);
vec2 shd_sq = sin_half_delta * sin_half_delta;
float a = shd_sq.y + cos(sourceRadians.y) * cos(targetRadians.y) * shd_sq.x;
return 2.0 * asin(sqrt(a));
}
vec3 interpolateGreatCircle(vec3 source, vec3 target, vec3 source3D, vec3 target3D, float angularDist, float t) {
vec2 lngLat;
if(abs(angularDist - PI) < 0.001) {
lngLat = (1.0 - t) * source.xy + t * target.xy;
} else {
float a = sin((1.0 - t) * angularDist);
float b = sin(t * angularDist);
vec3 p = source3D.yxz * a + target3D.yxz * b;
lngLat = degrees(vec2(atan(p.y, -p.x), atan(p.z, length(p.xy))));
}
float z = paraboloid(angularDist * EARTH_RADIUS, source.z, target.z, t);
return vec3(lngLat, z);
}
void main(void) {
geometry.worldPosition = instanceSourcePositions;
geometry.worldPositionAlt = instanceTargetPositions;
float segmentIndex = float(gl_VertexID / 2);
float segmentSide = mod(float(gl_VertexID), 2.) == 0. ? -1. : 1.;
float segmentRatio = getSegmentRatio(segmentIndex);
float prevSegmentRatio = getSegmentRatio(max(0.0, segmentIndex - 1.0));
float nextSegmentRatio = getSegmentRatio(min(arc.numSegments - 1.0, segmentIndex + 1.0));
float indexDir = mix(-1.0, 1.0, step(segmentIndex, 0.0));
isValid = 1.0;
uv = vec2(segmentRatio, segmentSide);
geometry.uv = uv;
geometry.pickingColor = picking_getPickingColorFromInstanceID();
vec4 curr;
vec4 next;
vec3 source;
vec3 target;
if ((arc.greatCircle || project.projectionMode == PROJECTION_MODE_GLOBE) && project.coordinateSystem == COORDINATE_SYSTEM_LNGLAT) {
source = project_globe_(vec3(instanceSourcePositions.xy, 0.0));
target = project_globe_(vec3(instanceTargetPositions.xy, 0.0));
float angularDist = getAngularDist(instanceSourcePositions.xy, instanceTargetPositions.xy);
vec3 prevPos = interpolateGreatCircle(instanceSourcePositions, instanceTargetPositions, source, target, angularDist, prevSegmentRatio);
vec3 currPos = interpolateGreatCircle(instanceSourcePositions, instanceTargetPositions, source, target, angularDist, segmentRatio);
vec3 nextPos = interpolateGreatCircle(instanceSourcePositions, instanceTargetPositions, source, target, angularDist, nextSegmentRatio);
if (abs(currPos.x - prevPos.x) > 180.0) {
indexDir = -1.0;
isValid = 0.0;
} else if (abs(currPos.x - nextPos.x) > 180.0) {
indexDir = 1.0;
isValid = 0.0;
}
nextPos = indexDir < 0.0 ? prevPos : nextPos;
nextSegmentRatio = indexDir < 0.0 ? prevSegmentRatio : nextSegmentRatio;
if (isValid == 0.0) {
nextPos.x += nextPos.x > 0.0 ? -360.0 : 360.0;
float t = ((currPos.x > 0.0 ? 180.0 : -180.0) - currPos.x) / (nextPos.x - currPos.x);
currPos = mix(currPos, nextPos, t);
segmentRatio = mix(segmentRatio, nextSegmentRatio, t);
}
vec3 currPos64Low = mix(instanceSourcePositions64Low, instanceTargetPositions64Low, segmentRatio);
vec3 nextPos64Low = mix(instanceSourcePositions64Low, instanceTargetPositions64Low, nextSegmentRatio);
curr = project_position_to_clipspace(currPos, currPos64Low, vec3(0.0), geometry.position);
next = project_position_to_clipspace(nextPos, nextPos64Low, vec3(0.0));
} else {
vec3 source_world = instanceSourcePositions;
vec3 target_world = instanceTargetPositions;
if (arc.useShortestPath) {
source_world.x = mod(source_world.x + 180., 360.0) - 180.;
target_world.x = mod(target_world.x + 180., 360.0) - 180.;
float deltaLng = target_world.x - source_world.x;
if (deltaLng > 180.) target_world.x -= 360.;
if (deltaLng < -180.) source_world.x -= 360.;
}
source = project_position(source_world, instanceSourcePositions64Low);
target = project_position(target_world, instanceTargetPositions64Low);
float antiMeridianX = 0.0;
if (arc.useShortestPath) {
if (project.projectionMode == PROJECTION_MODE_WEB_MERCATOR_AUTO_OFFSET) {
antiMeridianX = -(project.coordinateOrigin.x + 180.) / 360. * TILE_SIZE;
}
float thresholdRatio = (antiMeridianX - source.x) / (target.x - source.x);
if (prevSegmentRatio <= thresholdRatio && nextSegmentRatio > thresholdRatio) {
isValid = 0.0;
indexDir = sign(segmentRatio - thresholdRatio);
segmentRatio = thresholdRatio;
}
}
nextSegmentRatio = indexDir < 0.0 ? prevSegmentRatio : nextSegmentRatio;
vec3 currPos = interpolateFlat(source, target, segmentRatio);
vec3 nextPos = interpolateFlat(source, target, nextSegmentRatio);
if (arc.useShortestPath) {
if (nextPos.x < antiMeridianX) {
currPos.x += TILE_SIZE;
nextPos.x += TILE_SIZE;
}
}
curr = project_common_position_to_clipspace(vec4(currPos, 1.0));
next = project_common_position_to_clipspace(vec4(nextPos, 1.0));
geometry.position = vec4(currPos, 1.0);
}
float widthPixels = clamp(
project_size_to_pixel(instanceWidths * arc.widthScale, arc.widthUnits),
arc.widthMinPixels, arc.widthMaxPixels
);
vec3 offset = vec3(
getExtrusionOffset((next.xy - curr.xy) * indexDir, segmentSide, widthPixels),
0.0);
DECKGL_FILTER_SIZE(offset, geometry);
#ifdef ANTIALIASING
float halfWidthPixels = length(offset.xy);
if (halfWidthPixels > 0.0) {
float coverageScale = 1.0 + 0.5 / project.devicePixelRatio / halfWidthPixels;
offset.xy *= coverageScale;
uv.y *= coverageScale;
}
geometry.uv = uv;
#endif
DECKGL_FILTER_GL_POSITION(curr, geometry);
gl_Position = curr + vec4(project_pixel_size_to_clipspace(offset.xy), 0.0, 0.0);
vec4 color = mix(instanceSourceColors, instanceTargetColors, segmentRatio);
vColor = vec4(color.rgb, color.a * layer.opacity);
DECKGL_FILTER_COLOR(vColor, geometry);
}
`,k=`#version 300 es
#define SHADER_NAME arc-layer-fragment-shader
precision highp float;
in vec4 vColor;
in vec2 uv;
in float isValid;
out vec4 fragColor;
void main(void) {
#ifdef ANTIALIASING
float edgeCoord = abs(uv.y);
float edgePixels = (1.0 - edgeCoord) / max(fwidth(edgeCoord), 1e-6);
#endif
if (isValid == 0.0) {
discard;
}
#ifdef ANTIALIASING
if (edgePixels <= -SMOOTH_EDGE_RADIUS) {
discard;
}
#endif
fragColor = vColor;
geometry.uv = uv;
#ifdef ANTIALIASING
fragColor.a *= smoothedge(0.0, edgePixels);
#endif
DECKGL_FILTER_COLOR(fragColor, geometry);
}
`,A=[0,0,0,255],j={getSourcePosition:{type:`accessor`,value:e=>e.sourcePosition},getTargetPosition:{type:`accessor`,value:e=>e.targetPosition},getSourceColor:{type:`accessor`,value:A},getTargetColor:{type:`accessor`,value:A},getWidth:{type:`accessor`,value:1},getHeight:{type:`accessor`,value:1},getTilt:{type:`accessor`,value:0},greatCircle:!1,numSegments:{type:`number`,value:50,min:1},widthUnits:`pixels`,widthScale:{type:`number`,value:1,min:0},widthMinPixels:{type:`number`,value:0,min:0},widthMaxPixels:{type:`number`,value:2**53-1,min:0},antialiasing:!1},M=class extends v{getBounds(){return this.getAttributeManager()?.getBounds([`instanceSourcePositions`,`instanceTargetPositions`])}getShaders(){let{antialiasing:e}=this.props;return super.getShaders({vs:O,fs:k,source:D,defines:e?{ANTIALIASING:1}:{},modules:[g,h,p,E]})}get wrapLongitude(){return!1}initializeState(){this.getAttributeManager().addInstanced({instanceSourcePositions:{size:3,type:`float64`,fp64:this.use64bitPositions(),transition:!0,accessor:`getSourcePosition`},instanceTargetPositions:{size:3,type:`float64`,fp64:this.use64bitPositions(),transition:!0,accessor:`getTargetPosition`},instanceSourceColors:{size:this.props.colorFormat.length,type:`unorm8`,transition:!0,accessor:`getSourceColor`,defaultValue:A},instanceTargetColors:{size:this.props.colorFormat.length,type:`unorm8`,transition:!0,accessor:`getTargetColor`,defaultValue:A},instanceWidths:{size:1,transition:!0,accessor:`getWidth`,defaultValue:1},instanceHeights:{size:1,transition:!0,accessor:`getHeight`,defaultValue:1},instanceTilts:{size:1,transition:!0,accessor:`getTilt`,defaultValue:0}})}updateState(e){super.updateState(e);let{props:t,oldProps:n,changeFlags:r}=e;(r.extensionsChanged||t.antialiasing!==n.antialiasing)&&(this.state.model?.destroy(),this.state.model=this._getModel(),this.getAttributeManager().invalidateAll())}draw({uniforms:e}){let{widthUnits:t,widthScale:n,widthMinPixels:r,widthMaxPixels:i,greatCircle:a,wrapLongitude:o,numSegments:s}=this.props,c={numSegments:s,widthUnits:m[t],widthScale:n,widthMinPixels:r,widthMaxPixels:i,greatCircle:a,useShortestPath:o},l=this.state.model;l.shaderInputs.setProps({arc:c}),l.setVertexCount(s*2),l.draw(this.context.renderPass)}_getModel(){return new _(this.context.device,{...this.getShaders(),id:this.props.id,bufferLayout:this.getAttributeManager().getBufferLayouts(),topology:`triangle-strip`,isInstanced:!0})}};M.layerName=`ArcLayer`,M.defaultProps=j;var N=e(t(),1),P=`layout(std140) uniform particleTraceUniforms {
  float currentTime;
  float darknessScale;
} particleTrace;`,F={name:`particleTrace`,vs:P,fs:P,uniformTypes:{currentTime:`f32`,darknessScale:`f32`}},I=class extends M{static layerName=`ParticleTraceLayer`;static defaultProps={...M.defaultProps,currentTime:0,darknessScale:12,getTraceParameters:{type:`accessor`,value:[0,0,0,0]}};getShaders(){let e=super.getShaders();return e.modules=[...e.modules,F],e.inject={"vs:#decl":`in vec4 instanceTraceParameters; flat out vec4 vTraceParameters;`,"vs:#main-end":`vTraceParameters = instanceTraceParameters;`,"fs:#decl":`flat in vec4 vTraceParameters; float traceWeight;`,"fs:#main-start":`
        float progress = clamp(uv.x, 0.0, 1.0);
        float crossingTime = mix(vTraceParameters.x, vTraceParameters.y, progress);
        traceWeight = vTraceParameters.w + (particleTrace.currentTime >= crossingTime ? vTraceParameters.z : 0.0);
        // Unvisited segments are absent from depth and picking passes too.
        if (traceWeight <= 0.0) discard;
      `,"fs:DECKGL_FILTER_COLOR":`
        // Include antialias coverage inside the exponent so combining old
        // particles into a baseline preserves both the center and curve edges.
        // Every actual departure contributes only behind its own particle.
        color.a = 1.0 - exp(-traceWeight * color.a / particleTrace.darknessScale);
      `},e}initializeState(){super.initializeState(),this.getAttributeManager().addInstanced({instanceTraceParameters:{size:4,accessor:`getTraceParameters`}})}draw(e){let{currentTime:t,darknessScale:n}=this.props;this.state.model.shaderInputs.setProps({particleTrace:{currentTime:t,darknessScale:n}}),super.draw(e)}},L=n();function R({cells:e,replay:t,time:n,mode:p,onFailure:m,overlay:h=!1}){let g=(0,N.useRef)(null),_=(0,N.useRef)(null),[v,w]=(0,N.useState)({width:600,height:350}),[T,E]=(0,N.useState)(!1);(0,N.useEffect)(()=>{let e=new ResizeObserver(e=>{let t=e[0]?.contentRect;t?.width&&t.height&&w(e=>e.width===t.width&&e.height===t.height?e:{width:t.width,height:t.height})});return g.current&&e.observe(g.current),()=>e.disconnect()},[]);let D=(0,N.useMemo)(()=>S(e,v),[e,v]),O=(0,N.useMemo)(()=>new b({...v,...D}),[D,v]),k=(0,N.useMemo)(()=>new Map,[e,t,O]),A=(0,N.useMemo)(()=>i(t,p),[t,p]),j=(0,N.useMemo)(()=>{let e=Array.from({length:96},()=>[]),n=0;for(let r of t.events)e[r.bin].push({...r,firstRecord:(A-n%A)%A}),n+=r.count;return e},[t,A]),M=(0,N.useMemo)(()=>{let e=t.routes.map(()=>new Uint32Array(97));for(let t of j)for(let n of t)e[n.route][n.bin+1]=Math.max(0,Math.ceil((n.count-n.firstRecord)/A))*A;for(let t of e)for(let e=1;e<=96;e++)t[e]+=t[e-1];return e},[t,j,A]),P=(0,N.useMemo)(()=>h||T?null:C(`overview-flow-basemap`,()=>E(!0)),[h,T]),F=(0,N.useMemo)(()=>e.flatMap(e=>e.rings.map(t=>({...e,polygon:t}))),[e]),R=(0,N.useMemo)(()=>new y({id:`overview-flow-spatial-grid`,data:F,coordinateSystem:x.LNGLAT,getPolygon:e=>e.polygon,filled:!0,stroked:!0,extruded:!1,getFillColor:[255,253,253,115],getLineColor:[188,153,158,65],getLineWidth:.35,lineWidthUnits:`pixels`,lineWidthMinPixels:.2,pickable:!1}),[F]),z=l(n),B=Math.max(0,Math.floor(n-2)),V=(0,N.useMemo)(()=>r(t,p),[t,p]),H=(0,N.useMemo)(()=>{let e=e=>j[e].flatMap(n=>{let r=[];for(let i=n.firstRecord;i<n.count;i+=A)r.push({route:t.routes[n.route],parameters:s(e,n.departures[i],p,A)});return r});if(p===`interval`)return{base:[],current:e(z)};let n=t.routes.flatMap((e,t)=>{let n=M[t][B];return n?[{route:e,parameters:[0,0,0,n]}]:[]}),r=[];for(let t=B;t<=z;t++)r.push(...e(t));return{base:n,current:r}},[j,t,A,p,z,B,M]),U=(0,N.useMemo)(()=>{let n=t.routes.length>5e3,r=(r,i)=>new I({id:`overview-flow-trace-${i}`,data:r,coordinateSystem:x.LNGLAT,getSourcePosition:({route:t})=>[...e[t.origin].center,0],getTargetPosition:({route:t})=>[...e[t.destination].center,0],getSourceColor:[184,5,26,255],getTargetColor:[184,5,26,255],getWidth:n?.85:1.15,widthUnits:`pixels`,widthMinPixels:.35,widthMaxPixels:2.4,getHeight:({route:n})=>c(n,e,Math.max(1,t.maximum)),getTilt:14,greatCircle:!1,numSegments:64,pickable:!0,autoHighlight:!1,highlightColor:[109,0,17,230],antialiasing:!0,parameters:{depthWriteEnabled:!1},getTraceParameters:e=>e.parameters,currentTime:0,darknessScale:Math.max(5,Math.sqrt(V)*1.4)});return{base:r(H.base,`base`),current:r(H.current,`current`)}},[e,t,H,V,p,A,z]);return(0,N.useEffect)(()=>{let r=_.current;if(!r)return;let i=r.getContext(`2d`);if(!i){m();return}let o=Math.min(2,window.devicePixelRatio||1),s=Math.round(v.width*o),c=Math.round(v.height*o);if((r.width!==s||r.height!==c)&&(r.width=s,r.height=c),i.setTransform(o,0,0,o,0,0),i.clearRect(0,0,v.width,v.height),r.dataset.particleScale=String(A),r.dataset.replayTime=n.toFixed(3),r.dataset.flowMode=p,r.dataset.flowBin=String(l(n)),r.dataset.traceMode=`particle-drawn`,r.dataset.departureTiming=`recorded-seconds`,p===`accumulation`&&(n<=0||n>=96)){r.dataset.particleCount=`0`;return}let f=n=>{let r=k.get(n);return r||(r=a(t.routes[n],e,Math.max(1,t.maximum),O),k.set(n,r)),r},h=[],g=p===`interval`?l(n):Math.max(0,Math.floor(n-2));for(let e=g;e<=l(n);e++)for(let t of j[e])for(let r=t.firstRecord;r<t.count;r+=A){let[i,a]=u(e,t.departures[r],p);if(i>n)break;if(a<=n)continue;let o=(n-i)/(a-i),s=f(t.route),[c,l]=d(s,o);h.push({curve:s,progress:o,x:c,y:l})}r.dataset.particleCount=String(h.length);let y=(e,t,n)=>{let[r,a]=d(e,t);i.moveTo(r,a);for(let r=Math.floor(t*48)+1;r<=Math.floor(n*48);r++)i.lineTo(e[r*2],e[r*2+1]);let[o,s]=d(e,n);i.lineTo(o,s)};i.beginPath();for(let e of h)y(e.curve,Math.max(0,e.progress-.045),e.progress);i.strokeStyle=`rgba(183, 0, 24, .38)`,i.lineWidth=.8,i.stroke(),i.beginPath();for(let e of h)i.moveTo(e.x+1,e.y),i.arc(e.x,e.y,1,0,Math.PI*2);i.fillStyle=`rgba(175, 0, 22, .9)`,i.fill(),i.beginPath();for(let e of h)i.moveTo(e.x+.32,e.y),i.arc(e.x,e.y,.32,0,Math.PI*2);i.fillStyle=`rgba(255, 232, 232, .94)`,i.fill()},[e,k,j,p,m,A,t,v,n,O]),(0,L.jsxs)(`div`,{className:`city-deck-map`,ref:g,style:h?{background:`transparent`}:void 0,children:[(0,L.jsx)(f,{viewState:D,controller:!1,layers:[P,h?null:R,U.base.clone({currentTime:n,visible:p===`accumulation`}),U.current.clone({currentTime:n})],onError:()=>(m(),!0),getTooltip:({object:t})=>{if(!t||!(`route`in t))return null;let{route:r}=t,i=o(r,n,p);return i?{text:`Unit ${e[r.origin].id} → Unit ${e[r.destination].id}\n${i.toLocaleString(`en-US`)} ${p===`interval`?`trip records in this 15-minute interval`:`accumulated trip records`}\n${r.count.toLocaleString(`en-US`)} records over this 24-hour day\nCurves and travel speed are illustrative.`,style:{background:`#ffffffed`,color:`#542c34`,fontSize:`11px`,lineHeight:`1.5`,padding:`10px`,borderRadius:`7px`,maxWidth:`250px`}}:null}}),(0,L.jsx)(`canvas`,{ref:_,"aria-hidden":`true`,style:{position:`absolute`,inset:0,width:`100%`,height:`100%`,pointerEvents:`none`}}),!h&&(0,L.jsx)(`div`,{className:`city-preview-credit`,children:T?`Spatial grid · basemap unavailable`:(0,L.jsxs)(L.Fragment,{children:[`© `,(0,L.jsx)(`a`,{href:`https://www.openstreetmap.org/copyright`,target:`_blank`,rel:`noreferrer`,children:`OpenStreetMap`}),` contributors`]})})]})}var z=(0,N.memo)(R);export{z as default};