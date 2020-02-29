use std::path::Path;
use image;

pub fn open_image(filepath : &str) -> Option<Img> {
    let im = image::open(&Path::new(filepath)).unwrap().to_rgb();
    return Some(Img::new(im))
}

pub fn output_segmentations(edges : Vec<i32>, original_image : Img, filepath : &str) {
    // TODO
}

pub struct Img {
    height : usize,
    width : usize,
    length : usize,
    pixels : Vec<Pix>,
}

impl Img {
    fn new(data : image::RgbImage) -> Img {
        let height = data.height() as usize;
        let width = data.width() as usize;
        let mut pixels : Vec<Pix> = Vec::new();
        let tmp = data.into_raw();
        println!("Data has {} values", tmp.len());
        for i in 0..tmp.len()/3 {
            let i2 = 3*i;
            pixels.push(Pix::new(tmp[i2], tmp[i2+1], tmp[i2+2]));
        }
        return Img {height, width, length : height*width, pixels}
    }

    pub fn length(&self) -> usize {
        return self.length
    }

    pub fn dist_to_adj(&self, curr_v : usize, dir : i32) -> Option<(usize, f64)> {
        let (x, y) = self.idx_to_xy(curr_v);
        let (x2, y2) = match self.neighbor(x, y, dir) {
            None => return None,
            Some((p1, p2)) => (p1, p2), // TODO faire que ça marche c'est la 2mer
        };
        return Some((self.xy_to_idx(x2, y2), self.euclid_dist(x, x2, y, y2)))
    }

    pub fn get_opp_dir(dir : i32) -> i32 {
        return match dir {
            0 => 0,
            1 => 3,
            2 => 4,
            3 => 1,
            4 => 2,
            _ => {println!("Shouldn't happen"); return 0},
        }
    }

    fn idx_to_xy(&self, idx : usize) -> (usize, usize) {
        return (idx % self.width, idx/self.width)
    }

    fn xy_to_idx(&self, x : usize, y : usize) -> usize {
        return x + (self.width*y)
    }

    fn is_in_bounds(&self, x : usize, y : usize) -> bool {
        return x < self.width && y < self.height
    }

    fn neighbor(&self, x : usize, y : usize, dir : i32) -> Option<(usize, usize)> {
        let ib = self.is_in_bounds(x, y);
        let neigh_ib = match dir {
            0 => ib,
            1 => ib && y > 0,
            2 => x + 1 < self.width,
            3 => ib && y + 1 < self.height,
            4 => ib && x > 0,
            _ => {println!("Shouldn't happen"); ib},
        };
        if !neigh_ib {
            return None
        }
        else {
            return match dir {
                0 => Some((x, y)),
                1 => Some((x, y - 1)),
                2 => Some((x + 1, y)),
                3 => Some((x, y + 1)),
                4 => Some((x - 1, y)),
                _ => {println!("Shouldn't happen"); None},
            }
        }
    }

    fn euclid_dist(&self, x1 : usize, x2 : usize, y1 : usize, y2 : usize) -> f64 {
        return self.pixels[self.xy_to_idx(x1, y1)].dist(&self.pixels[self.xy_to_idx(x2, y2)])
    }
}

struct Pix {
    r : u8,
    g : u8,
    b : u8,
}

impl Pix {
    fn new(r : u8, g : u8, b : u8) -> Pix {
        return Pix{r, g, b}
    }

    fn dist(&self, other : &Pix) -> f64 {
        return ((Self::abs(self.r, other.r).pow(2) + Self::abs(self.g, other.g).pow(2) + Self::abs(self.b, other.b).pow(2)) as f64).sqrt() as u32
    }

    fn abs(c1 : u8, c2 : u8) -> f64 {
        return if c1 > c2 {(c1-c2) as f64} else {(c2-c1) as f64};
    }
}