const POP_SIZE : usize = 50;
const TOURNAMENT_SIZE : usize = 4;
const GENERATIONS : usize = 10;

const MUT_PROB : f64 = 0.1;
const CROSS_PROB : f64 = 1.0;

const OD_WEIGHT : f64 = 0.00525;
const CO_WEIGHT : f64 = 0.9945;
const EV_WEIGHT : f64 = 0.00025;

use crate::image_proc::Img;
use crate::image_proc::Pix;
use crate::b_heap::BinaryHeap;
use std::collections::HashSet;
use std::cmp::Ordering;
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
        let mut new_indiv : HashSet<Genome> = HashSet::new(); 
        while new_indiv.len() < POP_SIZE {
            let p1 = tournament_select(&pop);
            let p2 = tournament_select(&pop);
            let c1 = ((&pop[p1]).crossover(input_image, &pop[p2])).mutate();
            let c2 = ((&pop[p2]).crossover(input_image, &pop[p1])).mutate();
            new_indiv.insert(c1);
            new_indiv.insert(c2);
        }
        for new_p in new_indiv {
            pop.push(new_p);
        }
        pop.sort_by(|a, b| match a.fitness.partial_cmp(&b.fitness) {None => Ordering::Equal, Some(eq) => eq});
        pop.drain(0..POP_SIZE);
        println!("Best fitness : {}", pop[pop.len() - 1].fitness);
    }
    let best = pop.pop().unwrap();
    let segs = Genome::find_segments(input_image, &best.edges).0;
    return segs;
}


#[derive(Hash)]
#[derive(PartialEq)]
#[derive(Eq)]
struct Genome {
    fitness : i32,
    edges : Vec<i32>,
}

impl Genome {
    fn new(fitness : i32, edges : Vec<i32>) -> Genome {
        return Genome {fitness, edges};
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
        return Genome::new(self.fitness, self.edges.clone())
    }

    fn get_fitness(img : &Img, edges : &Vec<i32>) -> i32 {
        let (seg_nums, centroids) = Self::find_segments(img, edges);
        let (edge_val, connectivity, overall_dev) = Self::get_measures(img, &seg_nums, &centroids);
        println!("e {}, c {}, o {}", edge_val, connectivity, overall_dev);
        println!("{} segments", centroids.len());
        let fit = edge_val as f64 * EV_WEIGHT - connectivity as f64 * CO_WEIGHT - overall_dev as f64 * OD_WEIGHT;
        return fit as i32
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