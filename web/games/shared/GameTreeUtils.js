import { MarkerType } from 'reactflow';

// --- Configuration ---
const NODE_WIDTH = 40;
const NODE_HEIGHT = 40;
const X_SPACING = 60;  // Horizontal distance between moves
const Y_SPACING = 50;  // Vertical distance between variations

/**
 * Main function to convert raw history into React Flow elements
 * @param {Array} history - Array of 'state_update' payloads
 * @param {Function} renderFn - The seatToHtml function from GameAppBase
 */
export function getLayoutElements(history, renderFn) {
  if (!history || history.length === 0) return { nodes: [], edges: [] };

  // 1. Build the logical tree structure
  const treeMap = buildTreeStructure(history);
  console.log('Tree Map:', treeMap);

  // 2. Calculate coordinates (x, y)
  const { nodes, edges } = calculateCoordinates(treeMap, renderFn);

  return { nodes, edges };
}

/**
 * Step 1: Convert flat list to Tree Map
 * Keys are C++ game_tree_index (stringified)
 */
function buildTreeStructure(history) {
  const nodeMap = new Map();

  // Create a virtual root.
  // IMPORTANT: Ensure your C++ engine sends updates that link back to 0 or
  // whatever your root index is.
  nodeMap.set("0", {
    id: "0",
    children: [],
    depth: 0,
    data: { label: "Start", seat: null, moveNumber: 0 }
  });

  history.forEach((msg) => {
    const id = String(msg.game_tree_index);
    const parentId = String(msg.node_before_action);

    // Safety check: ensure parent exists.
    // If messages arrive out of order, this might skip nodes.
    // (Your bridge sort logic handles this, so it should be fine)
    if (!nodeMap.has(parentId)) return;

    const parent = nodeMap.get(parentId);

    const newNode = {
      id,
      parentId,
      children: [],
      depth: parent.depth + 1,
      // Store the raw C++ payload in 'data'
      data: {
        ...msg,
        moveNumber: parent.depth + 1
      }
    };

    nodeMap.set(id, newNode);
    parent.children.push(id);
  });

  return nodeMap;
}

/**
 * Step 2: Calculate X/Y positions
 * Logic: First child stays on same Y (Main Line), others drop down.
 */
function calculateCoordinates(nodeMap, renderFn) {
  const nodes = [];
  const edges = [];

  // Track used vertical space per column (depth) to prevent overlapping
  const nextYAtDepth = new Map();

  function traverse(nodeId, inheritedY) {
    const node = nodeMap.get(nodeId);
    if (!node) return;

    // --- X Position ---
    const x = node.depth * X_SPACING;

    // --- Y Position (Lane Logic) ---
    // If inheritedY is -1, we must find a new empty lane.
    // If inheritedY is valid, we try to stay there (main line).
    let y = inheritedY;

    // Check what Y is available at this depth
    const nextAvailableY = nextYAtDepth.get(node.depth) || 0;

    if (y === -1 || y < nextAvailableY) {
       // We were forced down, or our inherited lane is already taken
       y = nextAvailableY;
    }

    // Update the "High Water Mark" for this column
    nextYAtDepth.set(node.depth, y + Y_SPACING);

    // Create React Flow Node
    nodes.push({
      id: node.id,
      position: { x, y },
      type: 'gameNode', // Matches the key in GameTreePanel nodeTypes
      data: {
        ...node.data,
        seat: node.data.seat,
        renderFn: renderFn // Pass the visualization function
      },
      style: {
        width: NODE_WIDTH,
        height: NODE_HEIGHT,
      }
    });

    // Create React Flow Edge (if not root)
    if (node.parentId) {
      edges.push({
        id: `e${node.parentId}-${node.id}`,
        source: node.parentId,
        target: node.id,
        type: 'smoothstep',
        markerEnd: { type: MarkerType.ArrowClosed },
        style: { stroke: '#b1b1b7' }
      });
    }

    // --- Recurse ---
    if (node.children.length > 0) {
      // Child 0 is the "Main Line" -> Inherit current Y
      traverse(node.children[0], y);

      // Children 1..N are "Variations" -> Pass -1 to force new lane
      for (let i = 1; i < node.children.length; i++) {
        traverse(node.children[i], -1);
      }
    }
  }

  // Start traversal from Root
  traverse("0", 0);

  return { nodes, edges };
}
