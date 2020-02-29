pub struct BinaryHeap {
    node_list : Vec<HeapNode>,
    vertex_pos : Vec<usize>,
}

#[derive(Clone)]
struct HeapNode {
    key : f64,
    v_number : usize,
    edge_direction : i32,
}

// TODO : replace all by swap_elem and make sure vertex_pos is always up to date
impl BinaryHeap {
    pub fn new() -> BinaryHeap {
        return BinaryHeap {node_list : Vec::new(), vertex_pos : Vec::new()}
    }

    fn left_child(idx : usize) -> usize {
        return 2*idx + 1
    }

    fn right_child(idx : usize) -> usize {
        return 2*idx + 2
    }

    fn parent(idx : usize) -> usize {
        return (idx+1)/2 - 1
    }

    fn decrease_key(&mut self, mut idx : usize, new_key : f64) {
        self.node_list[idx].key = new_key;
        let mut p_key = Self::parent(idx);
        while new_key > self.node_list[p_key].key && idx > 0 {
            self.swap_elems(idx, p_key);
            idx = p_key;
            p_key = Self::parent(idx);
        }
    }

    pub fn insert(&mut self, v_number : usize, key : f64, direction : i32) {
        self.node_list.push(HeapNode{key, v_number, edge_direction : direction});
        self.decrease_key(self.node_list.len() - 1, key);
    }

    fn max_heapify(&mut self, mut idx : usize) {
        let mut over = false;
        while !over {
            let min_child = self.min_child(idx);
            if self.node_list[min_child].key < self.node_list[idx].key {
                self.swap_elems(idx, min_child)
            }
            else {
                over = true;
            }
            if Self::left_child(idx) >= self.node_list.len() {
                over = true;
            }
            idx = min_child;
        }
    }

    pub fn max_heapify_all(&mut self) {
        for i in (0..self.node_list.len()-1).rev() {
            self.max_heapify(i);
        }
    }

    fn min_child(&self, idx : usize) -> usize {
        let lc = Self::left_child(idx);
        let rc = Self::right_child(idx);
        if rc >= self.node_list.len() {
            return lc
        }
        else if self.node_list[lc].key > self.node_list[rc].key {
            return rc
        }
        else {
            return lc
        }
    }

    pub fn extract_max(&mut self) -> (f64, usize, i32) {
        self.swap_elems(0, self.node_list.len() - 1);
        let min = self.node_list.pop().unwrap();
        self.max_heapify(0);
        return (min.key, min.v_number, min.edge_direction)
    }

    pub fn try_update_smallest_edge(&mut self, v_num : usize, new_key : f64, new_dir : i32) {
        if new_key < self.node_list[self.vertex_pos[v_num]].key {
            self.decrease_key(self.vertex_pos[v_num], new_key);
            self.set_dir(v_num, new_dir);
        }
    }

    fn swap_elems(&mut self, pos1 : usize, pos2 : usize) {
        let v1 = self.node_list[pos1].v_number;
        let v2 = self.node_list[pos2].v_number;
        self.vertex_pos[v1] = pos2;
        self.vertex_pos[v2] = pos1;
        let tmp_k = self.node_list[pos1].clone();
        self.node_list[pos1] = self.node_list[pos2].clone();
        self.node_list[pos2] = tmp_k;
    }

    pub fn find_vertices(&mut self) {
        for (i, node) in self.node_list.iter().enumerate() {
            self.vertex_pos[node.v_number] = i;
        }
    }

    pub fn set_dir(&mut self, v_num : usize, new_dir : i32) {
        self.node_list[self.vertex_pos[v_num]].edge_direction = new_dir;
    }

    pub fn is_empty(&self) -> bool {
        return self.node_list.is_empty()
    }
}