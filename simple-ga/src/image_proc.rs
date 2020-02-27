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
    height : u32,
    width : u32,
    length : u32,
    pixels : Vec<Pix>,
}

impl Img {
    fn new(data : image::RgbImage) -> Img {
        let height = data.height();
        let width = data.width();
        let mut pixels : Vec<Pix> = Vec::new();
        let tmp = data.into_raw();
        println!("Data has {} values", tmp.len());
        for i in 0..tmp.len()/3 {
            let i2 = 3*i;
            pixels.push(Pix::new(tmp[i2], tmp[i2+1], tmp[i2+2]));
        }
        return Img {height, width, length : height*width, pixels}
    }

    pub fn length(&self) -> u32 {
        return self.length
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
}