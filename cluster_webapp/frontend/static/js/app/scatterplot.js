

//  Scatterplot Module
var ScatterPlot = function (dataset, catNum, selector) {
    /*
    Data should be in form:
        [ {x: <INT>, y: <INT>, c: <CATEGORY> }]
    */
    var cats = d3.nest()
        .key(function (d) { return d.c; })
        .entries(dataset);

    catNum = catNum || cats.length;

    //////////
    console.log(cats);
    console.log(catNum);
    //////////

    selector = selector || 'div.scatterplot svg.chart';
    var colors = d3.scale.category20();

    //  Chart variables
    var margin = {top: 10, right: 10, bottom: 20, left: 10};
    var size = {width: 700, height: 500};
    var padding = 40;
    //  Select DOM element
    var svg = d3.select(selector)
        .attr("width", size.width + margin.left + margin.right)
        .attr("height", size.height + margin.top + margin.bottom);
    //  Setup scales
    var xMin = d3.min(dataset, function (d) {return d.x;});
    var xMax = d3.max(dataset, function (d) { return d.x;});
    var xLim = Math.max(Math.abs(xMin), Math.abs(xMax));
    var xScale = d3.scale.linear()
        .domain([-xLim, +xLim])
        .range([padding, size.width - (padding * 2)]);
    var yMin = d3.min(dataset, function (d) {return d.y;});
    var yMax = d3.max(dataset, function (d) { return d.y;});
    var yLim = Math.max(Math.abs(yMin), Math.abs(yMax));
    var yScale = d3.scale.linear()
        .domain([-yLim, +yLim])
        .range([size.height - padding, padding]);
    var rScale = d3.scale.linear()
        .domain([-yLim, +yLim])
        .range([2, 5]);
    //  Setup axes
    //      X Axis
    var xAxis = d3.svg.axis()
        .scale(xScale)
        .orient("bottom")
        .ticks(10); //   Set rough # of ticks;
    //      Y Axis
    var yAxis = d3.svg.axis()
        .scale(yScale)
        .orient("left")
        .ticks(10); //   Set rough # of ticks;
    //  Define clipping path
    svg.append("clipPath")
        .attr("id", "chart-area")
        .append("svg:rect")
        .attr({
            x       : padding,
            y       : padding,
            width   : size.width,
            height  : size.height
        });
    //  Bind data
    svg.append('svg:g')
        .attr({
            id          : "circles",
            'clip-path' : "url(#chart-area)"
        })
        .selectAll('circle')
            .data(dataset)
            .enter()
                .append('svg:circle')
                .attr('cx', function (d) {
                    return xScale(d.x);
                })
                .attr('cy', function (d) {
                    return yScale(d.y);
                })
                .attr('r', function (d) {
                    return 5;
                })
                .attr('fill', function (d) {
                    var c = d.c;
                    if (String(c)[0] === 'c') {
                        //  Example: "c2_137418"
                        c = c.slice(1).split('_');
                        c = parseInt(c);
                    }
                    c = colors(c);
                    return c;
                });
    // Legend
    var legend = svg.selectAll("g.legend")
        .data(cats)
        .enter()
            .append("svg:g")
            .attr({
                "class"     : "legend",
                "transform" : function (d, i) {
                    return "translate(" + padding + "," + ((i * 20)  + size.height - padding - (catNum * 10)) + ")";
                }
            });
    legend.append("svg:circle")
        .attr({
            "class" : String,
            "r"     : 5,
            "fill"      : function (d, i) {
                var c = d.key;
                if (String(c)[0] === 'c') {
                    //  Example: "c2_137418"
                    c = c.slice(1).split('_');
                    c = parseInt(c);
                }
                return colors(c);
            }
        });
    legend.append("svg:text")
        .attr({
            x   : 12,
            dy  : ".31em",
        })
        .text( function (d) {
            var c = d.key;
            return "Cluster " + c;
        });
    //  Display axes
    svg.append("svg:g")
        .attr("class", "x axis")
        // Move to the center
        .attr("transform", "translate(0," + ((size.height - padding) / 2) + ")")
        .call(xAxis);
    svg.append("svg:g")
        .attr("class", "y axis")
        .attr("transform", "translate(" + ((size.width - padding) / 2) + ",0)")
        .call(yAxis);
    //      Add axis event handlers
    svg.select(".x.axis")
        .transition()
        .duration(1000)
        .call(xAxis)
    svg.select(".y.axis")
        .transition()
        .duration(1000)
        .call(yAxis)
};
