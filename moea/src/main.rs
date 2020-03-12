mod ga;
mod image_proc;

const FILEPATH1 : &str = "C://Users//Moi//Desktop//Desktop//EPFL//Bachelor//BA6//IT3708-Bio-Inspired-Artificial-Intelligence//Project 2//Training//";
const FILEPATH2 : &str = "//Test image.jpg";

fn main() {
    // let im_numbers : Vec<u32> = vec![86016, 118035, 147091, 176035, 176039, 216066, 353013];
    let im_numbers : Vec<u32> = vec![12074, 42044, 76002, 101087];
    // const CHOSEN_IMAGE : usize = 0;
    // let filepath_num = im_numbers[CHOSEN_IMAGE].to_string();
    // let img = image_proc::open_image(format!("{}{}{}", FILEPATH1, filepath_num, FILEPATH2).as_str()).unwrap();
    // let pfront = ga::train(&img);
    // image_proc::output_segmentations(img, pfront, filepath_num);
    for s in im_numbers {
        let filepath_num = s.to_string();
        let img = image_proc::open_image(format!("{}{}{}", FILEPATH1, filepath_num, FILEPATH2).as_str()).unwrap();
        let pfront = ga::train(&img);
        image_proc::output_segmentations(img, pfront, filepath_num);
    }
}
