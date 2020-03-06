const POP_SIZE : usize = 50;
const TOURNAMENT_SIZE : usize = 4;
const GENERATIONS : usize = 100;
const MIN_SEG_SIZE : usize = 100;

const MUT_PROB : f64 = 0.7;
const CROSS_PROB : f64 = 1.0;

use crate::image_proc::Img;
use crate::image_proc::Pix;
use crate::b_heap::BinaryHeap;
use std::collections::HashSet;
use rand::prelude::*;
use std::cmp::Ordering;
use std::f64::MAX as MAX_F64;
use std::i32::MAX as MAX_I32;
use rayon::prelude::*;
use rayon::iter::once;

// Directions for representation : 0 = None, 1 = Up, 2 = Right, 3 = Down, 4 = Left
pub fn train(input_image : &Img) -> Vec<Vec<usize>> {
    let mut pop : Vec<Genome> = Vec::new();
    while pop.len() < POP_SIZE {
        pop.push(Genome::random(input_image));
    }
    for i in 0..GENERATIONS {
        println!("Gen {}", i+1);
        let new_pop: HashSet<Genome> = (0..POP_SIZE/2).into_par_iter().flat_map(|_| {
            let p1 = tournament_select(&pop);
            let p2 = tournament_select(&pop);
            let c1 = ((&pop[p1]).crossover_order1(input_image, &pop[p2])).mutate(input_image);
            let c2 = ((&pop[p2]).crossover_order1(input_image, &pop[p1])).mutate(input_image);
            once(c1).chain(once(c2))
        }).collect();
        pop.extend(new_pop.into_iter());
        pop = rank_crowding_sort(pop);
        pop.drain(0..POP_SIZE);
    }
    let bests = get_pareto_front(pop);
    let segs : Vec<Vec<usize>> = bests.into_iter().map(|g| Genome::find_segments(input_image, &g.edges).0).collect();
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
            vec_dist_heap.insert(v as usize, MAX_F64, 0);
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

        return Genome::new(Self::get_fitness(img, &mut rd_edges), rd_edges)
    }

    fn mutate(mut self, img : &Img) -> Genome {
        let mut rng = thread_rng();
        let tmp : f64 = rng.gen();
        if tmp < MUT_PROB {
            let idx : usize = rng.gen_range(0, self.edges.len());
            let new_val : i32 = rng.gen_range(0, 5);
            self.edges[idx] = new_val;
        }
        return Self::new(Self::get_fitness(img, &mut self.edges), self.edges)
    }

    fn crossover_uniform(&self, img : &Img, other : &Genome) -> Genome {
        let mut rng = thread_rng();
        let tmp : f64 = rng.gen();
        if tmp < CROSS_PROB {
            let rd_num : Vec<f64> = (0..self.edges.len()).map(|_| rng.gen()).collect();
            let mut new_vec : Vec<i32> = rd_num.into_iter().enumerate().map(|(idx, f)| if f < 0.5 {self.edges[idx]} else {other.edges[idx]}).collect();
            let fitness = Self::get_fitness(img, &mut new_vec);
            return Genome::new(fitness, new_vec)
        }
        return (*self).clone()
    }

    fn crossover_order1(&self, img : &Img, other : &Genome) -> Genome {
        let mut rng = thread_rng();
        let tmp : f64 = rng.gen();
        if tmp < CROSS_PROB {
            let mut new_vec = Vec::new();
            let begin = rng.gen_range(0, self.edges.len());
            let length = rng.gen_range(0, self.edges.len() - begin);
            for &n in other.edges.iter().take(begin).chain(self.edges.iter().skip(begin).take(length).chain(other.edges.iter().skip(begin + length))) {
                new_vec.push(n);
            }
            let fitness = Self::get_fitness(img, &mut new_vec);
            return Genome::new(fitness, new_vec)
        }
        return (*self).clone()
    }

    fn get_fitness(img : &Img, edges : &mut Vec<i32>) -> (i32, i32, i32) {
        let (seg_nums, centroids) = Self::find_segments(img, edges);
        let (edge_val, connectivity, overall_dev) = Self::get_measures(img, &seg_nums, &centroids);
        // println!("{} segments", centroids.len());
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
            let mut border = Self::add_span(v, &mut untreated, &mut next_seg, &adj_list).0;
            while next_seg.len() < MIN_SEG_SIZE {
                let mut new = 0;
                for d in 1..=4 {
                    match img.neighbor(border, d) {
                        Some(n) => if !next_seg.contains(&n) {new = n},
                        None => (),
                    }
                }
                if untreated.contains(&new) {
                    adj_list[border].push(new);
                    adj_list[new].push(border);
                    border = Self::add_span(new, &mut untreated, &mut next_seg, &adj_list).0;
                }
                else {
                    let mut seg_num = 0;
                    for (i, old_seg) in segments.iter().enumerate() {
                        if old_seg.contains(&new) {
                            seg_num = i;
                        }
                    }
                    let to_merge = segments.remove(seg_num);
                    centr = centroid_sums.remove(seg_num);
                    next_seg.extend(to_merge);
                }
            }
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

    fn add_span(v : usize, untreated : &mut HashSet<usize>, treated : &mut HashSet<usize>, adj_list : &Vec<Vec<usize>>) -> (usize, usize) {
        if treated.contains(&v) {return (v, adj_list[v].len())}
        untreated.remove(&v);
        treated.insert(v);
        let (mut vert, mut least_n) = (v, adj_list[v].len());
        for &down_v in &adj_list[v] {
            let tmp = Self::add_span(down_v, untreated, treated, adj_list);
            if tmp.1 < least_n {
                least_n = tmp.1;
                vert = tmp.0;
            }
        }
        return (vert, least_n)
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

    fn dominated_by(&self, other : &Genome) -> bool {
        let mut at_least = true;
        let mut better = false;
        if other.edge_value > self.edge_value 
            || other.connectivity < self.connectivity 
            || other.overall_dev < self.overall_dev {
            better = true;
        }
        if other.edge_value < self.edge_value 
            || other.connectivity > self.connectivity 
            || other.overall_dev > self.overall_dev {
            at_least = false;
        }
        return at_least && better
    }
}

fn tournament_select(pop : &Vec<Genome>) -> usize {
    let mut rng = thread_rng();
    let mut candidates : Vec<usize> = (0..TOURNAMENT_SIZE).map(|_| rng.gen_range(0, pop.len())).collect();
    candidates.sort();
    return candidates.pop().unwrap()
}

fn rank_crowding_sort(mut pop : Vec<Genome>) -> Vec<Genome> {
    let mut ranks : Vec<usize> = Vec::new();
    let mut max_rank = 0;
    for e1 in &pop {
        let mut rank = 0;
        for e2 in &pop {
            if e1.dominated_by(e2) {rank = rank + 1;}
        }
        if rank > max_rank {max_rank = rank;}
        ranks.push(rank);
    }
    let mut by_rank : Vec<Vec<Genome>> = vec![Vec::new(); max_rank + 1];
    while !pop.is_empty() {
        by_rank[ranks.pop().unwrap()].push(pop.pop().unwrap());
    }
    while !by_rank.is_empty() {
        let r = sort_by_crowding(by_rank.pop().unwrap());
        pop.extend(r);
    }
    println!("Fronts : {}", max_rank+1);
    return pop
}

fn get_pareto_front(pop : Vec<Genome>) -> Vec<Genome> {
    let mut front : Vec<Genome> = Vec::new();
    for e1 in &pop {
        let mut rank = 0;
        for e2 in &pop {
            if e1.dominated_by(e2) {rank = rank + 1;}
        }
        if rank == 0 {
            front.push((*e1).clone())
        }
    }
    println!("Front size : {}", front.len());
    return pop
}

fn sort_by_crowding(subpop : Vec<Genome>) -> Vec<Genome> {
    println!("Front has length {}", subpop.len());
    if subpop.len() == 0 {return subpop}
    let mut sub : Vec<(Genome, f64)> = Vec::new();
    // Finding span for the 3 measures
    let (min_e, max_e, min_o, max_o, min_c, max_c) = subpop.iter().fold((0, MAX_I32, 0, MAX_I32, 0, MAX_I32),  
            |(mut min_e, mut max_e, mut min_o, mut max_o, mut min_c, mut max_c), g| {
                if g.edge_value > max_e {max_e = g.edge_value;}
                if g.edge_value < min_e {min_e = g.edge_value;}
                if g.overall_dev > max_o {max_o = g.overall_dev;}
                if g.overall_dev < min_o {min_o = g.overall_dev;}
                if g.connectivity > max_c {max_c = g.connectivity;}
                if g.connectivity < min_c {min_c = g.connectivity;}
                (min_e, max_e, min_o, max_o, min_c, max_c)
            });
    let (span_e, span_o, span_c) = ((max_e - min_e) as f64, (max_o - min_o) as f64, (max_c - min_c) as f64);
    
    for g in subpop {
        sub.push((g, 0.0));
    }
    let len = sub.len();

    // Sorting and adding distance values for edge value
    sub.sort_by(|a, b| a.0.edge_value.cmp(&b.0.edge_value));
    sub[0].1 = sub[0].1 + (MAX_F64/10.0);
    sub[len-1].1 = sub[len-1].1 + (MAX_F64/10.0);
    for i in 1..len-1 {
        let add_d = ((sub[i+1].0.edge_value - sub[i-1].0.edge_value) as f64)/span_e;
        sub[i].1 = sub[i].1 + add_d;
    }

    // Sorting and adding distance values for overall deviation
    sub.sort_by(|a, b| a.0.overall_dev.cmp(&b.0.overall_dev));
    sub[0].1 = sub[0].1 + (MAX_F64/10.0);
    sub[len-1].1 = sub[len-1].1 + (MAX_F64/10.0);
    for i in 1..len-1 {
        let add_d = ((sub[i+1].0.overall_dev - sub[i-1].0.overall_dev) as f64)/span_o;
        sub[i].1 = sub[i].1 + add_d;
    }

    // Sorting and adding distance values for connectivity
    sub.sort_by(|a, b| a.0.connectivity.cmp(&b.0.connectivity));
    sub[0].1 = sub[0].1 + (MAX_F64/10.0);
    sub[len-1].1 = sub[len-1].1 + (MAX_F64/10.0);
    for i in 1..len-1 {
        let add_d = ((sub[i+1].0.connectivity - sub[i-1].0.connectivity) as f64)/span_c;
        sub[i].1 = sub[i].1 + add_d;
    }

    //  Sort the vector by distance (biggest is best)
    sub.sort_by(|a, b| match a.1.partial_cmp(&b.1) {None => Ordering::Equal, Some(eq) => eq,});
    return sub.into_iter().map(|(g, _)| g).collect()
}