const POP_SIZE : usize = 50;
const TOURNAMENT_SIZE : usize = 4;
const GENERATIONS : usize = 100;

const MUT_PROB : f64 = 0.1;
const CROSS_PROB : f64 = 1.0;

use crate::image_proc::Img;
use crate::image_proc::Pix;
use crate::b_heap::BinaryHeap;
use std::collections::HashSet;
use rand::prelude::*;
use std::f64::MAX;

// Directions for representation : 0 = None, 1 = Up, 2 = Right, 3 = Down, 4 = Left
pub fn train(input_image : &Img) -> Vec<usize> {
    let mut pop : Vec<Genome> = Vec::new();
    while pop.len() < POP_SIZE {
        pop.push(Genome::random(input_image));
    }
    for i in 0..GENERATIONS {
        println!("Gen {}", i+1);
        let pool = pop.clone();
        let mut new_pop : HashSet<Genome> = pop.into_iter().collect();
        while new_pop.len() < 2*POP_SIZE {
            let p1 = tournament_select(&pool);
            let p2 = tournament_select(&pool);
            let c1 = ((&pool[p1]).crossover(input_image, &pool[p2])).mutate();
            let c2 = ((&pool[p2]).crossover(input_image, &pool[p1])).mutate();
            new_pop.insert(c1);
            new_pop.insert(c2);
        }
        pop = new_pop.into_iter().collect();
        rank_crowding_sort(&mut pop);
        pop.drain(0..POP_SIZE);
    }
    let best = pop.pop().unwrap();
    let segs = Genome::find_segments(input_image, &best.edges).0;
    return segs;
}


#[derive(Hash)]
#[derive(PartialEq)]
#[derive(Eq)]
#[derive(Clone)]
struct Genome {
    edge_value : i32,
    connectivity : i32,
    overall_dev : i32,
    edges : Vec<i32>,
}

impl Genome {
    fn new(measures : (i32, i32, i32), edges : Vec<i32>) -> Genome {
        return Genome {edge_value : measures.0, connectivity : measures.1, overall_dev : measures.2, edges};
    }

    fn random(img : &Img) -> Genome {
        let mut rd_edges : Vec<i32> = vec![0; img.length()]; 
        let mut rng = thread_rng();
        let mut vec_dist_heap = BinaryHeap::new();
        let start : usize = rng.gen_range(0, img.length()) as usize;
        for v in 0..img.length() {
            vec_dist_heap.insert(v as usize, MAX, 0);
        }
        vec_dist_heap.find_vertices();
        vec_dist_heap.try_update_smallest_edge(start, 0.0, 0);

        while !vec_dist_heap.is_empty() {
            let (dist, curr_v, dir) = vec_dist_heap.extract_min();
            rd_edges[curr_v] = dir;
            for d in 1..=4 {
                match img.dist_to_adj(curr_v, d) {
                    Some((adj_v, dist_to_adj)) => vec_dist_heap.try_update_smallest_edge(adj_v, dist + dist_to_adj, Img::get_opp_dir(d)),
                    None => (),
                }
            }
        }

        return Genome::new(Self::get_fitness(img, &rd_edges), rd_edges)
    }

    fn mutate(mut self) -> Genome {
        let mut rng = thread_rng();
        let tmp : f64 = rng.gen();
        if tmp < MUT_PROB {
            let idx : usize = rng.gen_range(0, self.edges.len());
            let new_val : i32 = rng.gen_range(0, 5);
            self.edges[idx] = new_val;
        }
        return self
    }

    fn crossover(&self, img : &Img, other : &Genome) -> Genome {
        let mut rng = thread_rng();
        let tmp : f64 = rng.gen();
        if tmp < CROSS_PROB {
            let rd_num : Vec<f64> = (0..self.edges.len()).map(|_| rng.gen()).collect();
            let new_vec : Vec<i32> = rd_num.into_iter().enumerate().map(|(idx, f)| if f < 0.5 {self.edges[idx]} else {other.edges[idx]}).collect();
            let fitness = Self::get_fitness(img, &new_vec);
            return Genome::new(fitness, new_vec)
        }
        return (*self).clone()
    }

    fn get_fitness(img : &Img, edges : &Vec<i32>) -> (i32, i32, i32) {
        let (seg_nums, centroids) = Self::find_segments(img, edges);
        let (edge_val, connectivity, overall_dev) = Self::get_measures(img, &seg_nums, &centroids);
        println!("{} segments", centroids.len());
        return (edge_val as i32, connectivity as i32, overall_dev as i32)
    }

    fn find_segments(img : &Img, edges : &Vec<i32>) -> (Vec<usize>, Vec<Pix>) {
        let mut seg_num = vec![0; edges.len()];
        let mut segments : Vec<HashSet<usize>> = Vec::new();
        let mut untreated : HashSet<usize> = HashSet::new();
        let mut centroid_sums : Vec<(u32, u32, u32)> = Vec::new();
        let mut adj_list : Vec<Vec<usize>> = vec![Vec::new(); img.length()];
        for v in 0..img.length() {
            untreated.insert(v);
            match img.neighbor(v, edges[v]) {
                None => (),
                Some(n) => {
                    adj_list[v].push(n);
                    adj_list[n].push(v);
                },
            }
        }

        while !untreated.is_empty() {
            let &v = untreated.iter().next().unwrap();
            let mut next_seg = HashSet::new();
            let mut centr = (0, 0, 0);
            Self::add_span(v, &mut untreated, &mut next_seg, &adj_list);
            for &a in &next_seg {
                centr = img.get(a).add_to_centroid_sum(centr);
            }
            segments.push(next_seg);
            centroid_sums.push(centr);
        }
        
        for (i, seg) in segments.iter().enumerate() {
            for &v in seg {
                seg_num[v] = i;
            }
        }

        let mut centroids = Vec::new();
        for i in 0..centroid_sums.len() {
            let (r_sum, g_sum, b_sum) = centroid_sums[i];
            let num_pixels = segments[i].len() as u32;
            centroids.push(Pix::new((r_sum/num_pixels) as u8, (g_sum/num_pixels) as u8, (b_sum/num_pixels) as u8));
        }
        return (seg_num, centroids)
    }

    fn add_span(v : usize, untreated : &mut HashSet<usize>, treated : &mut HashSet<usize>, adj_list : &Vec<Vec<usize>>) {
        if treated.contains(&v) {return}
        untreated.remove(&v);
        treated.insert(v);
        for &down_v in &adj_list[v] {
            Self::add_span(down_v, untreated, treated, adj_list);
        }
    }

    fn get_measures(img : &Img, seg_nums : &Vec<usize>, centroids : &Vec<Pix>) -> (f64, f64, f64) {
        let mut edge_val = 0.0;
        let mut connectivity = 0.0;
        let mut overall_dev = 0.0;
        for p in 0..img.length() {
            let p_pix = img.get(p);
            let seg = seg_nums[p];
            overall_dev = overall_dev + centroids[seg].dist(img.get(p));
            for d in 1..=8 {
                match img.neighbor(p, d) {
                    Some(n) => {
                        if seg_nums[n]!=seg {
                            edge_val = edge_val + p_pix.dist(img.get(n));
                            connectivity = connectivity + 0.125;
                        }
                    },
                    None => (),
                }
            }
        }
        return (edge_val, connectivity, overall_dev)
    }
}

fn tournament_select(pop : &Vec<Genome>) -> usize {
    let mut rng = thread_rng();
    let mut candidates : Vec<usize> = (0..TOURNAMENT_SIZE).map(|_| rng.gen_range(0, pop.len())).collect();
    candidates.sort();
    return candidates.pop().unwrap()
}

fn rank_crowding_sort(pop : &mut Vec<Genome>) {
    // TODO
    // sorting by rank : rank is the number of solutions it's dominated by
    // then make a vec for each rank, that you sort by crowding
    // p.71
    // Notes : crowding distance : p.85
}