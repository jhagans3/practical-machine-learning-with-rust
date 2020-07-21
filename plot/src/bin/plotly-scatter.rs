use plotly::common::Mode;
use plotly::plot::ImageFormat;
use plotly::{Plot, Scatter};

// https://igiagkiozis.github.io/plotly/content/getting_started.html
fn line_and_scatter_plot() {
    let trace1 = Scatter::new(vec![1, 2, 3, 4], vec![10, 15, 13, 17])
        .name("trace1")
        .mode(Mode::Markers);
    let trace2 = Scatter::new(vec![2, 3, 4, 5], vec![16, 5, 11, 9])
        .name("trace2")
        .mode(Mode::Lines);
    let trace3 = Scatter::new(vec![1, 2, 3, 4], vec![12, 9, 15, 12]).name("trace3");

    let mut plot = Plot::new();
    plot.add_trace(trace1);
    plot.add_trace(trace2);
    plot.add_trace(trace3);
    plot.show();

    // plot.show_png(1280, 900);
    // plot.save("plot_name.ext", ImageFormat::PNG, 1280, 900, 1.0);
}

fn main() -> std::io::Result<()> {
    line_and_scatter_plot();
    Ok(())
}
