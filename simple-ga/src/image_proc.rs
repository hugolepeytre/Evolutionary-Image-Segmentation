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
    signed_width : i32,
    length : usize,
    signed_length : i32,
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
        return Img {height, width, signed_width : width as i32, length : height*width, signed_length : (height*width) as i32, pixels}
    }

    pub fn length(&self) -> usize {
        return self.length
    }

    pub fn dist_to_adj(&self, curr_v : usize, dir : i32) -> Option<(usize, f64)> {
        let p2 = match self.neighbor(curr_v, dir) {
            None => return None,
            Some(o) => o,
        };
        return Some((p2, self.euclid_dist(curr_v, p2)))
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

    fn is_in_bounds(&self, p : i32) -> bool {
        return 0 <= p && p < self.length() as i32
    }

    pub fn neighbor(&self, p : usize, dir : i32) -> Option<usize> {
        let tmp_p = p as i32;
        let tmp_next = match dir {
            0 => tmp_p,
            1 => tmp_p - self.signed_width,
            2 => tmp_p + 1,
            3 => tmp_p + self.signed_width,
            4 => tmp_p - 1,
            5 => tmp_p - self.signed_width + 1,
            6 => tmp_p + self.signed_width + 1,
            7 => tmp_p + self.signed_width - 1,
            8 => tmp_p - self.signed_width - 1,
            _ => {println!("Shouldn't happen"); tmp_p},
        };
        let next = if self.is_in_bounds(tmp_next) {Some(tmp_next as usize)} else {None};
        return next
        
    }

    fn euclid_dist(&self, p1 : usize, p2 : usize) -> f64 {
        return self.pixels[p1].dist(&self.pixels[p2])
    }

    pub fn get(&self, idx : usize) -> &Pix {
        return &self.pixels[idx];
    }
}

pub struct Pix {
    r : u8,
    g : u8,
    b : u8,
}

impl Pix {
    fn new(r : u8, g : u8, b : u8) -> Pix {
        return Pix{r, g, b}
    }

    pub fn dist(&self, other : &Pix) -> f64 {
        return (Self::abs(self.r, other.r).powf(2.0) + Self::abs(self.g, other.g).powf(2.0) + Self::abs(self.b, other.b).powf(2.0)).sqrt()
    }

    fn abs(c1 : u8, c2 : u8) -> f64 {
        return if c1 > c2 {(c1-c2) as f64} else {(c2-c1) as f64};
    }
}