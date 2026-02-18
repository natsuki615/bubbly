// spatial Hash for efficient proximity queries
class SpatialHash {
    constructor(cellSize = 50) {
        this.cellSize = cellSize;
        this.grid = new Map(); // key: "x,y" -> value: array of particle indices
        this.particlePositions = []; // parallel array of particle positions
    }
    
    clear() {
        this.grid.clear();
        this.particlePositions = [];
    }
    
    // add a particle with its position
    insert(index, x, y) {
        this.particlePositions[index] = [x, y];
        
        const cellX = Math.floor(x / this.cellSize);
        const cellY = Math.floor(y / this.cellSize);
        const key = `${cellX},${cellY}`;
        
        if (!this.grid.has(key)) {
            this.grid.set(key, []);
        }
        this.grid.get(key).push(index);
    }
    
    // get all particles within a certain distance of a point
    getNearby(x, y, radius) {
        const results = [];
        
        const minCellX = Math.floor((x - radius) / this.cellSize);
        const maxCellX = Math.floor((x + radius) / this.cellSize);
        const minCellY = Math.floor((y - radius) / this.cellSize);
        const maxCellY = Math.floor((y + radius) / this.cellSize);
        
        for (let cellX = minCellX; cellX <= maxCellX; cellX++) {
            for (let cellY = minCellY; cellY <= maxCellY; cellY++) {
                const key = `${cellX},${cellY}`;
                const cellParticles = this.grid.get(key);
                
                if (cellParticles) {
                    for (let idx of cellParticles) {
                        const [px, py] = this.particlePositions[idx];
                        const dist = Math.hypot(px - x, py - y);
                        if (dist < radius) {
                            results.push({ index: idx, x: px, y: py, dist: dist });
                        }
                    }
                }
            }
        }
        
        return results;
    }
    
    // get particles in a specific cell
    getCell(x, y) {
        const cellX = Math.floor(x / this.cellSize);
        const cellY = Math.floor(y / this.cellSize);
        const key = `${cellX},${cellY}`;
        return this.grid.get(key) || [];
    }
}
