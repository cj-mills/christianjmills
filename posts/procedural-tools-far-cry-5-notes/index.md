---
categories:
- procedural-generation
- game-dev
- notes
date: 2021-12-9
description: "In this presentation from 2018, Etienne Carrier details the procedural pipeline developed for Far Cry 5 using Houdini and Houdini Engine, focusing on its objectives, tools, user workflow, underlying mechanics, and lessons learned during development. 
"
hide: false
search_exclude: false
title: "Notes on *Procedural World Generation of Ubisoft’s Far Cry 5*"

aliases:
- /Notes-on-the-Procedural-Tools-Used-to-Make-Far-Cry-5/


twitter-card:
  creator: "@cdotjdotmills"
  site: "@cdotjdotmills"
  image: /images/default-preview-image-black.png
open-graph:
  image: /images/default-preview-image-black.png

---



* [Introduction and Challenges](#introduction-and-challenges)
* [Objectives of the Procedural Pipeline](#objectives-of-the-procedural-pipeline)
* [Procedural Tools Developed](#procedural-tools-developed)
* [User Workflow: Filling an Empty Map](#user-workflow-filling-an-empty-map)
* [Under the Hood: Houdini Engine and Data Exchange](#under-the-hood-houdini-engine-and-data-exchange)
* [Cliff Tool: Detailed Breakdown](#cliff-tool-detailed-breakdown)  
* [Biome Tool: Populating the World with Life](#biome-tool-populating-the-world-with-life)  
* [Lessons Learned](#lessons-learned)
* [Conclusion](#conclusion)





::: {.callout-tip title="Source Material"}

* **Video:** [Procedural World Generation of Ubisoft’s Far Cry 5](https://www.youtube.com/watch?v=NfizT369g60)
* **Slides:** [Far Cry 5: Procedural World Generation](https://ubm-twvideo01.s3.amazonaws.com/o1/vault/gdc2018/presentations/ProceduralWorldGeneration.pdf)

:::



## Introduction and Challenges

* **Etienne Carrier**, Technical Artist at Ubisoft Montreal (3 years), presents the procedural pipeline developed for Far Cry 5.
* **Challenge:** Terrain undergoing constant changes during the project's 2.5-year development.
    * Manual content placement (e.g., forests) becomes incoherent with each terrain iteration.
    * Repainting content manually after each terrain change is tedious and difficult to maintain consistency across a large world with multiple users.
    * Locking terrain early is unrealistic as iterations are crucial for game quality.



## Objectives of the Procedural Pipeline

* Develop a **macro management tool** to:
    * Fill the world with natural-looking content.
    * Ensure content consistency with terrain topology, adapting to changes (e.g., forest distribution remaining coherent despite terrain adjustments).
* **Automation:**
    * Utilize Houdini, Houdini Engine, and nightly builds on build machines to fully refresh the world daily.
    * Build machines process different map sections, ensuring up-to-date world data each morning.
* **Deterministic Generation:**
    * Ensure the generation yields identical results with the same inputs, regardless of the build machine used.
    * This is crucial for seamless map junctions and nav mesh generation.
* **User-Friendliness:**
    * Provide in-editor tools for on-demand procedural generation alongside the nightly builds.



## Procedural Tools Developed

* Expanded beyond the initial mandate of biome distribution to include:
    * **Freshwater Tools:** Generate lakes, rivers, streams, and waterfalls.
    * **Fence and Power Line Tools:** Create fences and power lines along splines.
    * **Cliff Generation:** Generates cliffs on steep terrain.
    * **Biome Tool:** Spawns vegetation throughout the world.
    * **Fog Density Map Generation:** Creates a 2D map based on terrain topology and content placement to influence the fog shader.
    * **Wall Map Terrain Generation:** Generates a low-detail terrain representation for the world map, including miniature trees.



## User Workflow: Filling an Empty Map

1. **Terraforming:**
    * Initial world machine pass followed by artist-driven terraforming using in-editor tools.
2. **Freshwater Network:**
    * Artists lay down a network of curves and splines to define rivers and other water bodies.
    * Procedural generation creates the water surface, waterside assets, and terrain texturing based on spline parameters (e.g., width).
3. **Cliff Generation:**
    * Tool automatically generates cliffs based on terrain slope, with minimal user input.
    * Includes features like exclusion masks for finer control.
    * Often handled by nightly builds, relieving artists from frequent manual generation.
4. **Vegetation (Biome Painter):**
    * Artists paint sub-biomes (e.g., forest) using a painter tool.
    * Main biomes automatically distribute vegetation (grass, trees) based on terrain data and sub-biome information.
    * Recipes react to water proximity, altitude, and cliff erosion lines, spawning appropriate assets.
5. **Customization:**
    * Artists can override the natural distribution by clearing areas with grass sub-biome and adding elements like roads (using splines).
    * Road generation clears vegetation and adds roads, requiring some manual asset placement (houses, vehicles).
    * Biome painting can be further refined to create features like driveways using terrain textures.
    * Forest sub-biome can be repainted to add trees around specific locations.
6. **Non-Destructive Workflow:**
    * Terrain can be adjusted at any time, requiring a refresh of affected procedural tools (e.g., cliffs, biomes).
    * System maintains coherence and updates content placement automatically.
7. **Fence and Power Line Placement:**
    * Fences are generated along splines with fence type specified as an attribute.
    * Power lines connect electric poles placed at spline control points.
    * System supports multiple power line types and automatically handles snapping and transformer placement.
    * Biome tool automatically clears vegetation obstructing power lines.



## Under the Hood: Houdini Engine and Data Exchange

* **Houdini Engine** within Dunia enables seamless data exchange between Houdini and the engine.
* **Inputs from Dunia to Houdini (via Python script):**
    * World information (name, size).
    * File paths for assets and data.
    * Terrain sector information (area to generate).
    * Splines and shapes with metadata on geometry attributes (e.g., fence type on a fence spline).
* **Inputs from Disk (file paths provided by Dunia):**
    * Height maps.
    * Biome painter data.
    * SPNG and 2D terrain masks.
    * Houdini geometry from previously generated tools.
* **Primary Input:** The terrain itself, as generation is linked to specific terrain areas (**sectors**).
* **Terrain Subdivision:**
    * **Sectors** are the smallest unit (64x64 meters), defining the minimum area for procedural generation.
* **Baking Procedural Data:**
    * Users select a generation area:
        * All (everything loaded in the editor).
        * Local map section or sectors (under the camera).
        * Frustum (sectors visible from the camera).
* **Outputs from Houdini to Dunia:**
    * Entity point cloud (vegetation, rocks, collectibles, decals, VFX, etc.).
    * Terrain textures.
    * Terrain height maps.
    * 2D terrain data (RGB or grayscale).
    * Procedurally generated geometry.
    * Terrain logic zones (IDs for post-processing and fog presets).
* **Data Transfer:** Outputs are saved as buffers in a specific format for efficient loading by the editor, avoiding large memory transfers.
* **Tool Interconnectivity:**
    * Procedural generation is sequential (freshwater, cliffs, biomes, etc.).
    * Tools export data to influence subsequent tools (e.g., freshwater generates a water mask used by the biome tool).



## Cliff Tool: Detailed Breakdown

### Previous Tech and Motivation

* Far Cry 4 and Primal had no dedicated cliff tech, relying on terrain and placed rocks.
* Worlds were designed to minimize large cliff surfaces due to visual limitations and the cost of manual rock placement.
* Far Cry 5 featured larger and more prominent cliffs, necessitating a new approach.

### Cliff Tool Inputs and Geometry

* **Terrain slope** is the primary input.
* Surfaces below a defined slope threshold are deleted, creating the **cliff input geometry**.
* Cliffs serve as a visual cue for impassable terrain.
* Remeshing is applied to eliminate stretched quads on slopes, resulting in uniform triangles.

### Stratification

* **Geological stratification** (horizontal lines in sedimentary rock) is simulated.
* Tool slices the input geometry into **strata chunks** with random thickness.
* Each **strata** receives a unique ID for color variation (debug view).
* **Strata angle** is controlled by user-painted RGB input on the terrain, with four presets available in the editor.
* Noise is used to split the cliff surface into two groups, each stratified with a different seed to break up perfect lines and create more natural patterns.

### Geometry Generation and Export

* Strata are extruded with varying thickness and displaced using displacement maps.
* Triangle count reduction is applied for optimization.
* Geometry is split into individual meshes per sector (64x64 meters) for improved loading and streaming.

### Terrain Shading and Data Transfer

* Cliffs are shaded using the **terrain shader**, inheriting textures from the terrain below.
* To avoid texture mismatch, cliff textures are transferred to the terrain beneath the extruded cliff mesh.
* **Cliff mask** and **strata attributes** are raycast onto the terrain.
* **Strata attribute** generates a color layer for macro tint variation, even when cliff mesh is unloaded at a distance.
* **Cliff mask** is extended using a flow simulation to create an **erosion effect**, retaining the original strata color.
* **Crumbled rock entities** are scattered on the erosion area and exported as a point cloud.
* **Terrain texture IDs** are generated from the erosion mask, using noise to blend two cliff textures for tiling variation.

### Vegetation on Cliff Ledges

* Upward-facing polygons on cliffs are identified as potential vegetation areas.
* Raycasting checks for clearance above the surface.
* Trees and other vegetation are spawned on viable cliff ledges.

### Cliff Tool Exported Data

* Cliff geometry with collision information.
* Point cloud for rocks and ledge vegetation.
* Terrain texture IDs.
* Cliff color layer for the terrain.
* Cliff mask.



## Biome Tool: Populating the World with Life

### Initial Steps

* Terrain is generated from the height map.
* **Abiotic data** (physical features) is calculated:
    * Occlusion.
    * Flow map.
    * Slope.
    * Curvature.
    * Illumination.
    * Altitude, latitude, longitude.
    * Wind vector map.
* These attributes form the basis for biome recipes.
* 2D data from Dunia is imported:
    * Biome painting (user input).
    * Procedural data from previous tools (freshwater, roads, fences, power lines, cliff masks).

### Main Biomes and Sub-Biomes

* **Main biomes** cover most of the world (75-85%).
* They automatically determine **sub-biome** distribution (e.g., forest vs. grassland) based on abiotic terrain data, creating natural macro-level patterns.
* Main biomes also handle specific rules:
    * Replacing forest with grassland where power lines are present.

### Sub-Biome Recipes

* Sub-biomes are processed sequentially.
* Each **recipe** contains ingredients (trees, saplings, bushes, grass).
* Core of each recipe is the **generate-terrain-entities HDA** (Houdini Digital Asset), responsible for:
    * Vegetation scattering.
    * Terrain attribute modification.
    * Defining **viability** for each species.

### Viability and Species Competition

* **Viability** determines the likelihood of a species growing at a specific location, based on favored terrain attributes.
* Species with the highest accumulated viability at a location "wins."
* Example:
    * Species A favors a specific occlusion range (power of 1).
    * Species B favors a specific flow map range (power of 2).
    * Species B will dominate where the flow map value is sufficiently high.
* **Viability radius** determines the influence area of a species, preventing other species from growing too close.
* **Priority** and **priority radius** allow finer control over species interaction:
    * Priority is evaluated first.
    * If priorities are equal, viability is considered.
    * Example: Trees (priority 10) can have a smaller priority radius, allowing bushes (priority 0) to grow closer while still preventing overlap with other trees.

### Combining Terrain Attributes for Natural Patterns

* Mixing various terrain attributes allows for complex vegetation patterns and fluctuating viability.
* Example: Combining occlusion, altitude, flow map, and noise to create specific growth conditions.
* **Exclusion masks** from previous tools (water, cliffs, roads) can be incorporated.

### Controlling Asset Size

* **Asset size** is linked to viability, allowing for natural size variation within a species.
* Example: A conifer species can have multiple size variations (50m, 40m, 30m, etc.), each assigned to a specific viability range on the terrain.
* This creates a tapering effect at forest edges, with smaller trees at the periphery.
* **Scaling percentage** for each size prevents a staircase effect, allowing smooth transitions between sizes.
* **Random scale** can be added for further variation.
* Each size can have multiple **variations** (e.g., a dead tree variant), randomly selected with controllable weights.

### Forest Canopy and Ecological Succession

* **Age parameter** (signed distance field from viability data) influences size distribution, mimicking forest canopy and ecological succession.
* It can be added, multiplied, or interpolated with viability to control its influence.
* **Age maximum distance** controls the depth of the forest border tapering effect.
* Age ramp can be used to profile the overall forest shape.

### Scattering Density

* **Density ramp** controls the scattering density based on size, age, or viability.
* It prevents overlap issues with larger assets and manages performance by controlling asset numbers.
* Terrain attributes (illumination, slope aspect) can be mixed into the density ramp for finer control (e.g., reduced density on poorly lit slopes).

### Instance Tinting

* Biomes with high color variation are achieved by tinting individual instances.
* Gradient color ramps are driven by viability, age, or terrain data.
* Example: Water signed distance field drives grass color variation near water bodies.

### Instance Rotation

* Asset rotation is primarily based on terrain slope, aligning the forward axis with the slope.
* This allows for effects like grass leaning towards water due to the shore's slope.
* Other options include:
    * **Flat horizontal alignment.**
    * **Wind vector map alignment** (e.g., for wheatgrass).
    * **Rotation jitter and offset** on all axes for random variation.

### Terrain Modification by Scattered Assets

* Scattered assets can influence terrain properties.
* Four types of terrain data can be modified:
    * **Terrain deformation:** Height map adjustment (e.g., raising terrain around tree trunks).
    * **Terrain textures:** Applying specific textures underneath assets (e.g., tree roots).
    * **Terrain data output (mask):** Generating masks for reuse by other species (e.g., spawning rocks around specific trees).
    * **Terrain color:** Blending colors with terrain textures (e.g., simulating terrain humidity).
* Each data type has independent signed distance field settings for controlling the influence area.

### Terrain Deformation

* Masks generated from scattered assets can be combined with terrain data to control deformation.
* Example: Preventing terrain deformation near roads using the road mask.
* Displacement height deforms the terrain (e.g., raising it by 1 meter around trees).

### Terrain Textures

* Masks generated from scattered assets determine terrain texture application.
* Example: Applying a "root" texture underneath trees.
* Texture selection is dynamically fetched from the engine via Python script.
* Mask scale controls the area of influence for each texture.

### Terrain Data Output and Inter-Species Dependency

* Terrain data attributes (masks) can be generated from scattered assets and reused by other species.
* Example:
    * Ponderosa tree outputs a viability mask.
    * Forest rock species uses the Ponderosa viability mask to drive its spawning behavior, creating a dependency between the two.

### Terrain Color and Humidity Simulation

* **Terrain tint** is generated and mixed with organic terrain textures, similar to the cliff strata color.
* This creates color variations (e.g., dry brown to lush green) without increasing the number of terrain textures.
* Neutral gray color allows for both darkening and lightening of the terrain texture.
* Terrain color and texture variations can transpire through grass, controlled by a mask on the grass asset.

### Biome Tool Exported Data

* Entity point clouds.
* Terrain texture IDs.
* Terrain height map.
* Terrain color.
* Forest mask (used by fog and wall map tools).



## Lessons Learned

* **Responsibility:**
    * Procedural tools offer immense control over performance, gameplay, and art, requiring careful consideration of their impact.
    * Balancing art direction (e.g., dense forests) with gameplay needs (e.g., AI navigation) is crucial.
* **Elegant Design:**
    * Tools should open up possibilities and be adaptable for various scenarios.
    * The biome tool's flexibility is highlighted as an example.
* **Simplicity:**
    * Avoid over-engineering.
    * Strive for simple and elegant designs through iterative development and refinement.
* **User Feedback:**
    * Listen to user needs and preferences.
    * Sometimes manual control is preferred over automation (e.g., riverbed carving).
* **Flexibility:**
    * Adapt to changing requirements and be prepared to deviate from initial plans.
* **Balance:**
    * Find the right balance between control and automation.
    * Excessive automation can lead to issues, while excessive manual control can be time-consuming and difficult to manage.



## Conclusion

* The procedural pipeline significantly enhanced Far Cry 5's development, providing efficiency, control, and natural-looking environments.
* Houdini played a crucial role in achieving the project's goals.
* Collaboration and iteration were key to the pipeline's success.
* The modular and flexible design allows for future adaptation and reuse in other projects. 



 








{{< include /_about-author-cta.qmd >}}
