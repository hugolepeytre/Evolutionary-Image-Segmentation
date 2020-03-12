const POP_SIZE : usize = 50;
const TOURNAMENT_SIZE : usize = 4;
const GENERATIONS : usize = 200;
const _MIN_SEG_SIZE : usize = 5;
const FINAL_SAMPLE : usize = 20;
const MAX_SEG_NUM : i32 = 50;
const INIT_SEGS_SQRT : i32 = 7;
const MAX_SPREAD_SIZE : i32 = 50;

const MIN_RG_SEG_SIZE : usize = 200;
const MAX_RG_SEG_SIZE : usize = 300;

const MUT_PROB : f64 = 0.01;
const CROSS_PROB : f64 = 1.0;

use crate::image_proc::Img;
use crate::image_proc::Pix;
use std::collections::HashSet;
use rand::prelude::*;
use std::cmp::Ordering;
use std::f64::MAX as MAX_F64;
use std::i32::MAX as MAX_I32;
use rayon::prelude::*;
use rayon::iter::once;
use std::time::SystemTime;

// Directions for representation : 0 = None, 1 = Up, 2 = Right, 3 = Down, 4 = Left
pub fn train(input_image : &Img) -> Vec<Vec<usize>> {
    let beg = SystemTime::now();
    let mut now = SystemTime::now();
    let mut pop : Vec<Genome> = Vec::new();
    while pop.len() < POP_SIZE {
        pop.push(Genome::random(input_image));
        pop.push(Genome::random2(input_image));
    }
    for i in 0..GENERATIONS {
        println!("Gen {}", i+1);
        let new_pop: HashSet<Genome> = (0..POP_SIZE/2).into_par_iter().flat_map(|_| {
            let p1 = tournament_select(&pop);
            let p2 = tournament_select(&pop);
            let c1 = (&pop[p1]).cross_mut(input_image, &pop[p2]);
            let c2 = (&pop[p2]).cross_mut(input_image, &pop[p1]);
            once(c1).chain(once(c2))
        }).collect();
        pop.extend(new_pop.into_iter());
        pop = rank_crowding_sort(pop);
        pop = pop.drain((pop.len()-POP_SIZE)..).collect();
        if let Ok(dur) = now.elapsed() {
            println!("{}m{}s", dur.as_secs()/60, dur.as_secs()%60);
            now = SystemTime::now();
        }
    }
    let bests = get_pareto_front(pop);
    let mut segs : Vec<Vec<usize>> = bests.into_iter().map(|g| g.segmentation).collect();
    if let Ok(dur) = beg.elapsed() {
        println!("{}m{}s", dur.as_secs()/60, dur.as_secs()%60);
    }
    return segs.drain(0..FINAL_SAMPLE).collect();
}


#[derive(Hash)]
#[derive(PartialEq)]
#[derive(Eq)]
#[derive(Clone)]
struct Genome {
    avg_edge_value : i32,
    avg_overall_dev : i32,
    avg_seg_size : i32,
    segmentation : Vec<usize>,
    adj_list : Vec<Vec<usize>>,
    num_segs : i32,
}

impl Genome {
    fn new(measures : (i32, i32, i32, i32), segmentation : Vec<usize>, adj_list : Vec<Vec<usize>>) -> Genome {
        return Genome {avg_edge_value : measures.0, avg_overall_dev : measures.1, avg_seg_size : measures.2, segmentation, adj_list, num_segs : measures.3};
    }

    fn random(img : &Img) -> Genome {
        let mut rng = thread_rng();
        let num_segs : usize = rng.gen_range(3, INIT_SEGS_SQRT + 1) as usize;
        let mut rd_segm : Vec<usize> = vec![num_segs*num_segs; img.length()]; 
        let width = img.width() / num_segs;
        let height = img.height() / num_segs;
        for i in 0..num_segs*num_segs {
            let x = (i % num_segs) * width;
            let y = (i / num_segs) * height;
            for y2 in 0..height {
                for x2 in 0..width {
                    let new_y = y + y2;
                    let new_x = x + x2;
                    let pos = new_y * img.width() + new_x;
                    rd_segm[pos] = i;
                }
            }
        }
        let num_segs = num_segs*num_segs + 1;
        let adj_list = Self::make_adj_list(&rd_segm, img);
        return Genome::new(Self::get_fitness(img, &mut rd_segm, num_segs as i32), rd_segm, adj_list)
    }

    fn random2(img : &Img) -> Genome {
        let mut untreated : HashSet<usize> = (0..img.length()).collect();
        let mut rd_segm = vec![0; img.length()];
        let mut rng = thread_rng();
        let mut current_segment = 1;
        while !untreated.is_empty() {
            let next = *untreated.iter().next().unwrap();
            let seg_size = rng.gen_range(MIN_RG_SEG_SIZE, MAX_RG_SEG_SIZE);
            Self::make_rd_seg(next, current_segment, &mut rd_segm, seg_size, img, &mut untreated);
            current_segment = current_segment + 1;
        }
        println!("did one");
        let adj_list = Self::make_adj_list(&rd_segm, img);
        return Genome::new(Self::get_fitness(img, &mut rd_segm, current_segment as i32), rd_segm, adj_list)
    }

    fn make_rd_seg(current : usize, new_seg : usize, segmentation : &mut Vec<usize>, spread_size : usize, img : &Img, untreated : &mut HashSet<usize>) {
        if spread_size == 0 || segmentation[current] == new_seg { return }
        untreated.remove(&current);
        segmentation[current] = new_seg;
        for d in 1..=4 {
            if let Some(n) = img.neighbor(current, d) {
                Self::make_rd_seg(n, new_seg, segmentation, spread_size - 1, img, untreated);
            }
        }
    }

    // Each pixel can mutate. If it does, it goes to a random neighbor segment (need to work on a copy)
    fn _mutate(self, img : &Img) -> Genome {
        let mut rng = thread_rng();
        let rd_num : Vec<f64> = (0..img.length()).map(|_| rng.gen()).collect();
        let new_segmentation = rd_num.into_iter().enumerate().map(|(i, rd)| {
            if rd < MUT_PROB {
                let neigh = rng.gen_range(1, 5);
                match img.neighbor(i, neigh) {
                    Some(n) => self.segmentation[n],
                    None => self.segmentation[i]
                }
            }
            else {
                self.segmentation[i]
            }
        }).collect();
        let adj_list = Self::make_adj_list(&new_segmentation, img);
        let (new_segmentation, num_segs) = Self::renumber(&adj_list);
        return Self::new(Self::get_fitness(img, &new_segmentation, num_segs as i32), new_segmentation, adj_list)
    }

    fn renumber(adj_list : &Vec<Vec<usize>>) -> (Vec<usize>, usize) {
        let mut segmentation = vec![0; adj_list.len()];
        let mut untreated : HashSet<usize> = (0..adj_list.len()).collect();
        let mut curr_segment = 0;
        while !untreated.is_empty() {
            let next = *untreated.iter().next().unwrap();
            Self::renumber_rec(next, curr_segment, adj_list, &mut untreated, &mut segmentation);
            curr_segment = curr_segment + 1;
        }
        return (segmentation, curr_segment)
    }

    fn renumber_rec(next : usize, curr_segment : usize, adj_list : &Vec<Vec<usize>>, untreated : &mut HashSet<usize>, segmentation : &mut Vec<usize>) {
        let mut stack = Vec::new();
        stack.push(next);
        while !stack.is_empty() {
            let next = stack.pop().unwrap();
            segmentation[next] = curr_segment;
            untreated.remove(&next);
            for &neigh in &adj_list[next] {
                if untreated.contains(&neigh) {
                    stack.push(neigh);
                }
            }
        }
    }

    fn cross_mut(&self, img : &Img, other : &Genome) -> Genome {
        let mut rng = thread_rng();
        let tmp : f64 = rng.gen();
        let mut new_segmentation = vec![0; img.length()];
        // Crossover
        if tmp < CROSS_PROB {
            let mut untreated : HashSet<usize> = (0..img.length()).collect();
            let mut num_parent = 0;
            let mut curr_segment = 0;
            while !untreated.is_empty() {
                let next = *untreated.iter().next().unwrap();
                let used_adj = if num_parent == 0 { &self.adj_list } else { &other.adj_list };
                Self::renumber_rec(next, curr_segment, used_adj, &mut untreated, &mut new_segmentation);
                num_parent = 1 - num_parent;
                curr_segment = curr_segment + 1;
            }
        }
        else {
            new_segmentation = self.segmentation.clone();
        }
        // Mutate
        let rd_num : Vec<f64> = (0..img.length()).map(|_| rng.gen()).collect();
        for i in 0..new_segmentation.len() {
            if rd_num[i] < MUT_PROB {
                let neigh = rng.gen_range(1, 5);
                match img.neighbor(i, neigh) {
                    Some(n) => {
                        // if n's segment is different from i's, bomb
                        let s = new_segmentation[n];
                        if s != new_segmentation[i] {
                            let spread_size = rng.gen_range(1, MAX_SPREAD_SIZE);
                            Self::spread_flip(i, s, &mut new_segmentation, spread_size, img);
                        }
                    },
                    None => (),
                }
            }
        }
        // Make Genome
        let adj_list = Self::make_adj_list(&new_segmentation, img);
        let (new_segmentation, num_segs) = Self::renumber(&adj_list);
        return Self::new(Self::get_fitness(img, &new_segmentation, num_segs as i32), new_segmentation, adj_list)
    }

    fn spread_flip(current : usize, new_seg : usize, segmentation : &mut Vec<usize>, spread_size : i32, img : &Img) {
        if spread_size == 0 || segmentation[current] == new_seg { return }
        segmentation[current] = new_seg;
        for d in 1..=4 {
            if let Some(n) = img.neighbor(current, d) {
                Self::spread_flip(n, new_seg, segmentation, spread_size - 1, img);
            }
        }
    }

    // While selecting each parent alternatively, copy one of its segments containing a not yet assigned pixel into the child
    fn _crossover(&self, img : &Img, other : &Genome) -> Genome {
        let mut rng = thread_rng();
        let tmp : f64 = rng.gen();
        if tmp < CROSS_PROB {
            let mut new_segmentation = vec![0; img.length()];
            let mut untreated : HashSet<usize> = (0..img.length()).collect();
            let mut num_parent = 0;
            let mut curr_segment = 0;
            while !untreated.is_empty() {
                let next = *untreated.iter().next().unwrap();
                let used_adj = if num_parent == 0 { &self.adj_list } else { &other.adj_list };
                Self::renumber_rec(next, curr_segment, used_adj, &mut untreated, &mut new_segmentation);
                num_parent = 1 - num_parent;
                curr_segment = curr_segment + 1;
            }
            let adj_list = Self::make_adj_list(&new_segmentation, img);
            let (new_segmentation, num_segs) = Self::renumber(&adj_list);
            return Self::new(Self::get_fitness(img, &new_segmentation, num_segs as i32), new_segmentation, adj_list)
        }
        return (*self).clone()
    }

    fn get_fitness(img : &Img, segments : &Vec<usize>, num_segs : i32) -> (i32, i32, i32, i32) {
        let centroids = Self::get_centroids(img, segments, num_segs);
        let (avg_edge_val, avg_overall_dev) = Self::get_measures(img, segments, &centroids);
        return (avg_edge_val as i32/num_segs, avg_overall_dev as i32/num_segs, segments.len() as i32/num_segs, num_segs)
    }

    fn get_centroids(img : &Img, segments : &Vec<usize>, num_segs : i32) -> Vec<Pix> {
        let mut centroids = Vec::new();
        let mut sums : Vec<((u32, u32, u32), u32)> = vec![((0, 0, 0), 0); num_segs as usize];
        for (i, &seg) in segments.iter().enumerate() {
            sums[seg] = img.get(i).add_to_centroid_sum(sums[seg]);
        }
        for ((r, g, b), elems) in sums {
            if elems == 0 {
                centroids.push(Pix::new(0, 0, 0));
            }
            else {
                centroids.push(Pix::new((r/elems) as u8, (g/elems) as u8, (b/elems) as u8));
            }
        }
        return centroids
    }

    fn get_measures(img : &Img, seg_nums : &Vec<usize>, centroids : &Vec<Pix>) -> (f64, f64) {
        let mut edge_val = 0.0;
        let mut overall_dev = 0.0;
        for p in 0..img.length() {
            let p_pix = img.get(p);
            let seg = seg_nums[p];
            overall_dev = overall_dev + centroids[seg].dist(img.get(p));
            for d in 1..=4 {
                match img.neighbor(p, d) {
                    Some(n) => {
                        if seg_nums[n]!=seg {
                            edge_val = edge_val + p_pix.dist(img.get(n));
                        }
                    },
                    None => (),
                }
            }
        }
        return (edge_val, overall_dev)
    }

    fn make_adj_list(segmentation : &Vec<usize>, img : &Img) -> Vec<Vec<usize>> {
        let mut adj_list : Vec<Vec<usize>> = vec![Vec::new(); img.length()];
        for (i, &n) in segmentation.iter().enumerate() {
            for d in 1..=4 {
                if let Some(neigh) = img.neighbor(i, d) {
                    if n == segmentation[neigh] {
                        adj_list[i].push(neigh);
                    }
                }
            }
        }
        return adj_list
    }

    fn dominated_by(&self, other : &Genome) -> bool {
        let c1  = self.satisfies_constraints();
        let c2  = other.satisfies_constraints();
        if c1 && !c2 {
            return false
        }
        else if c2 && !c1 {
            return true
        }
        else if !c1 {
            return other.num_segs < self.num_segs
        }
        else {
            let mut at_least = true;
            let mut better = false;
            if other.avg_edge_value > self.avg_edge_value 
                || other.avg_seg_size > self.avg_seg_size 
                || other.avg_overall_dev < self.avg_overall_dev {
                better = true;
            }
            if other.avg_edge_value < self.avg_edge_value 
                || other.avg_seg_size < self.avg_seg_size 
                || other.avg_overall_dev > self.avg_overall_dev {
                at_least = false;
            }
            return at_least && better
        }
    }

    fn satisfies_constraints(&self) -> bool {
        return self.num_segs < MAX_SEG_NUM
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
                if g.avg_edge_value > max_e {max_e = g.avg_edge_value;}
                if g.avg_edge_value < min_e {min_e = g.avg_edge_value;}
                if g.avg_overall_dev > max_o {max_o = g.avg_overall_dev;}
                if g.avg_overall_dev < min_o {min_o = g.avg_overall_dev;}
                if g.avg_seg_size > max_c {max_c = g.avg_seg_size;}
                if g.avg_seg_size < min_c {min_c = g.avg_seg_size;}
                (min_e, max_e, min_o, max_o, min_c, max_c)
            });
    let (span_e, span_o, span_c) = ((max_e - min_e) as f64, (max_o - min_o) as f64, (max_c - min_c) as f64);
    
    for g in subpop {
        sub.push((g, 0.0));
    }
    let len = sub.len();

    // Sorting and adding distance values for edge value
    sub.sort_by(|a, b| a.0.avg_edge_value.cmp(&b.0.avg_edge_value));
    sub[0].1 = sub[0].1 + (MAX_F64/10.0);
    sub[len-1].1 = sub[len-1].1 + (MAX_F64/10.0);
    for i in 1..len-1 {
        let add_d = ((sub[i+1].0.avg_edge_value - sub[i-1].0.avg_edge_value) as f64)/span_e;
        sub[i].1 = sub[i].1 + add_d;
    }

    // Sorting and adding distance values for overall deviation
    sub.sort_by(|a, b| a.0.avg_overall_dev.cmp(&b.0.avg_overall_dev));
    sub[0].1 = sub[0].1 + (MAX_F64/10.0);
    sub[len-1].1 = sub[len-1].1 + (MAX_F64/10.0);
    for i in 1..len-1 {
        let add_d = ((sub[i+1].0.avg_overall_dev - sub[i-1].0.avg_overall_dev) as f64)/span_o;
        sub[i].1 = sub[i].1 + add_d;
    }

    // Sorting and adding distance values for connectivity
    sub.sort_by(|a, b| a.0.avg_seg_size.cmp(&b.0.avg_seg_size));
    sub[0].1 = sub[0].1 + (MAX_F64/10.0);
    sub[len-1].1 = sub[len-1].1 + (MAX_F64/10.0);
    for i in 1..len-1 {
        let add_d = ((sub[i+1].0.avg_seg_size - sub[i-1].0.avg_seg_size) as f64)/span_c;
        sub[i].1 = sub[i].1 + add_d;
    }

    //  Sort the vector by distance (biggest is best)
    sub.sort_by(|a, b| match a.1.partial_cmp(&b.1) {None => Ordering::Equal, Some(eq) => eq,});
    return sub.into_iter().map(|(g, _)| g).rev().collect()
}