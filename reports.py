"""
Created on Tue Oct  8 15:11:32 2019

Functions to produce html reports.

Alistair Curd
University of Leeds
2 October 2019

Software Engineering practices applied

Joanna Leng (an EPSRC funded Research Software Engineering Fellow (EP/R025819/1)
University of Leeds
October 2019

---
Copyright 2019 Peckham Lab

Licensed under the Apache License, Version 2.0 (the "License"); you may not use
this file except in compliance with the License. You may obtain a copy of the
License at
http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
"""

import numpy as np
import sys

def write_html_report_start(fout, info):
    '''Creates the start of an html report which is generic for the PERPL
    scripts. It does not create the file name or open the file.
    '''
    prog = info['prog']


    fout.write("<!DOCTYPE html>\n")
    fout.write("<html>\n")
    fout.write("<head>\n")
    fout.write("<meta charset=\"UTF-8\">\n")
    fout.write("<style>\n")
    fout.write("table, th, td {\n")
    fout.write("    border: 1px solid black;\n")
    fout.write("    border-collapse: collapse;\n")
    fout.write("}\n")
    fout.write("th, td {\n")
    fout.write("    padding: 15px;\n")
    fout.write("}\n")
    fout.write("</style>\n")

    title_line1 = ("<title>Report on *** Produced by +++ of the PERPL "
                   "Analysis Software</title>\n")
    title = title_line1.replace("***", info['in_file_no_path'])
    title = title.replace("+++", prog)
    fout.write(title)
    fout.write("</head>\n")
    fout.write("\n<body>\n")
    title_line2 = ("<h1 align=\"center\">Report on *** Produced by +++ "
                   "of the PERPL Analysis Software</h1>\n")
    title2 = title_line2.replace("***", info['in_file_no_path'])
    title2 = title2.replace("+++", prog)
    fout.write(title2)

    program_info = '<p><i>%s</i>: %s</p>\n' % (info['prog'], info['description'])
    fout.write(program_info)

    return fout




def write_html_report_end(fout):
    '''Ends and closes an html report which is generic for the PERPL
    scripts.
    '''
    fout.write("</body>\n")
    fout.write("</html>\n")

    fout.close()




def write_rot_2d_html_report(info, symmetries, aiccs, weights, table_values):
    """Creates and saves a html report. Contains images of graphs.

    Args:
        dims (int):
            The dimensions of the data ie 2D or 3D.
        info (dict):
            A python dictionary containing a collection of useful
            parameters such as the filenames and paths.

    Returns:
       Nothing is returned.
    """

    html_outfile_name = (info['results_dir']+r"/"+info['in_file_no_extension']
                         +r"_"+info['prog']+r"_report.html")
    if info['short_names'] is True:
        html_outfile_name = (info['short_results_dir']+r"/"
                             +info['short_filename_without_extension']+r"_"
                             +info['prog_short_name']+r"_report.html")

    try:
        fout = open(html_outfile_name, "w")
    except IOError as e:
        print("open() failed", e)
        print("The html report file could not be opened for writing. Try using"
              " the s flag to use shorter names.")
        sys.exit()

    fout = write_html_report_start(fout, info)


    fout.write(r"<p>This program ran at "+info['start']+r" on the "+info['host']
               +r" host system and the data file analysed was "
               +info['in_file_and_path']+r" which read in "+str(info['values'])
               +" relative positions with "+str(info['columns'])+" columns. "
               "This report provides images and information on simulated "
               "models of experimental fluorescence super-resolution light "
               "microscopy data and their comparison to experimental data.</p>\n"
               "The model "+info['model_name']+" is fitted to the relative "
               "positions, up to a maximum of "+str(info['filter_dist'])+" nm. "
               "The initial guesses for the parameter values were "
               +str(info['p0'])+" and the bounds on them during optimisation "
               "were "+str(info['optimisation_bounds'])+" .</p>\n")


    fout.write("<p></p>\n")
    fout.write("<h2>Curves of Best Fit of Simulated Data with a Histogram of "
               "Experimental Data</h2>\n")
    fout.write("<p>A histogram of separations between observations in experimental "
               "fluorescence localisation microscopy superimposed with curves "
               "of best fit for a variety of model data.</p>\n")

    fout.write("<IMG SRC=*** ALT=\"A histogram of separations between observations "
               "in experimental flurence super resolution light microscopy superimposed "
               "with curves of best fit for a variety of model data.\" height=100% "
               "width=100%>\n".replace("***", "Histogram_with_Fitted_Curves.png"))
    fout.write("<p></p>\n")
    fout.write("<h2>Data on Overall Model Fit</h2>\n")
    fout.write("<p>This table provides data on the fit of the simulated model "
               "data to experimental data. A lower corrected Akaike Information "
               "Criterion (AICc) for a model means that that model is more likely "
               "to be the closest to describing the truth underlying the data, "
               "among the models assessed. The Akaike weights for the models show "
               "these relative likelihoods, summing to one.</p>\n")

    fout.write("<table>\n")
    fout.write("   <tr>\n")
    fout.write("     <th>Symmetry</th>\n")
    fout.write("     <th>AICc</th>\n")
    fout.write("     <th>Akaike Weight</th>\n")
    fout.write("   </tr>\n")

    for i, symmetry in enumerate(symmetries):
        fout.write('   <tr> <td> %d </td> <td> %.2f </td> <td> %.2g '
                   '</td> </tr> \n' %(symmetry, aiccs[i], weights[i]))

    fout.write("</table>\n")
    fout.write("<p></p>\n")

    fout.write("<h2>Data on Model Fit for Each Model</h3>\n")

    fout.write("<p>Parametric model for distances between localisations on vertices "
               "of a polygon (order of symmetry = number of vertices). Includes "
               "the effect of localisation precision and unresolvable substructure "
               "at the polygon vertices.</p>\n")

    fout.write("<p>In this model, we take account of the fact that rotationally "
               "symmetric complexes may be unlikely to be found within one another: "
               "we allow the isotropic (linearly increasing) background term to "
               "remain zero until an onset distance (bgonset) is reached.</p>\n")

    fout.write("<h3>Table of Parameters</h3>\n")
    fout.write("<table>\n")
    fout.write("  <tr>\n")
    fout.write("    <th rowspan=\"2\">Parameter</th>\n")
    fout.write("    <th rowspan=\"2\">Definition</th>\n")
    fout.write("    <th colspan=\"3\">Settings used for the leaset squares curve fitting</th>\n")
    fout.write("  </tr>\n")
    fout.write("  <tr>\n")
    fout.write("    <th>Initial guess</th>\n")
    fout.write("    <th>Lower bounds</th>\n")
    fout.write("    <th>Upper bounds</th>\n")
    fout.write("  </tr>\n")
    fout.write("  <tr>\n")
    fout.write("  	<td>1</td>\n")
    fout.write("   	<td>Diameter of the circle containing the vertices of the "
               "polygon.</td>\n")
    fout.write("   	<td>"+str(info['p0'][0])+"</td>\n")
    fout.write("   	<td>"+str(info['optimisation_bounds'][0][0])+"</td>\n")
    fout.write("   	<td>"+str(info['optimisation_bounds'][1][0])+"</td>\n")
    fout.write("  <tr>\n")
    fout.write("	    <td>2</td>\n")
    fout.write("   	<td>Broadening of the peaks located at distances between "
               "the vertices.</td>\n")
    fout.write("   	<td>"+str(info['p0'][1])+"</td>\n")
    fout.write("   	<td>"+str(info['optimisation_bounds'][0][1])+"</td>\n")
    fout.write("   	<td>"+str(info['optimisation_bounds'][1][1])+"</td>\n")
    fout.write("  </tr>\n")
    fout.write("  <tr>\n")
    fout.write("  	<td>3</td>\n")
    fout.write(" 	<td>Amplitude of the contribution of one inter-vertex "
               "distance.</td>\n")
    fout.write("   	<td>"+str(info['p0'][2])+"</td>\n")
    fout.write("   	<td>"+str(info['optimisation_bounds'][0][2])+"</td>\n")
    fout.write("   	<td>"+str(info['optimisation_bounds'][1][2])+"</td>\n")
    fout.write("  </tr>\n")
    fout.write("  <tr>\n")
    fout.write(" 	<td>4</td>\n")
    fout.write(" 	<td>Spread representing the localisation precision for "
               "repeated localisations of the same fluorescent molecule.</td>\n")
    fout.write("   	<td>"+str(info['p0'][3])+"</td>\n")
    fout.write("   	<td>"+str(info['optimisation_bounds'][0][3])+"</td>\n")
    fout.write("   	<td>"+str(info['optimisation_bounds'][1][3])+"</td>\n")
    fout.write("  </tr>\n")
    fout.write("  <tr>\n")
    fout.write(" 	<td>5</td>")
    fout.write(" 	<td>Amplitude of the contribution of repeated localisations "
               "of the same fluorescent molecule.</td>")
    fout.write("   	<td>"+str(info['p0'][4])+"</td>\n")
    fout.write("   	<td>"+str(info['optimisation_bounds'][0][4])+"</td>\n")
    fout.write("   	<td>"+str(info['optimisation_bounds'][1][4])+"</td>\n")
    fout.write("  </tr>")
    fout.write("  <tr>")
    fout.write(" 	<td>6</td>\n")
    fout.write(" 	<td>Spread of a contribution resulting from unresolvable "
               "substructure, or mislocalisations resulting from a combination "
               "of simultaneous nearby emitters.</td>\n")
    fout.write("   	<td>"+str(info['p0'][5])+"</td>\n")
    fout.write("   	<td>"+str(info['optimisation_bounds'][0][5])+"</td>\n")
    fout.write("   	<td>"+str(info['optimisation_bounds'][1][5])+"</td>\n")
    fout.write("  </tr>\n")
    fout.write("  <tr>\n")
    fout.write(" 	<td>7</td>\n")
    fout.write(" 	<td>Amplitude of the contribution of unresolvable substructure, "
               "or mislocalisations resulting from a combination of simultaneous "
               "nearby emitters.</td>\n")
    fout.write("   	<td>"+str(info['p0'][6])+"</td>\n")
    fout.write("   	<td>"+str(info['optimisation_bounds'][0][6])+"</td>\n")
    fout.write("   	<td>"+str(info['optimisation_bounds'][1][6])+"</td>\n")
    fout.write("  </tr>\n")
    fout.write("  <tr>\n")
    fout.write(" 	<td>8</td>\n")
    fout.write(" 	<td>Gradient of an isotropic (linearly increasing) background "
               "term.</td>\n")
    fout.write("   	<td>"+str(info['p0'][7])+"</td>\n")
    fout.write("   	<td>"+str(info['optimisation_bounds'][0][7])+"</td>\n")
    fout.write("   	<td>"+str(info['optimisation_bounds'][1][7])+"</td>\n")
    fout.write("  </tr>\n")
    fout.write("  <tr>\n")
    fout.write(" 	<td>9</td>\n")
    fout.write(" 	<td>Onset distance for linearly increasing background term, "
               "since rotationally symmetric structures may exclude one another."
               "</td>\n")
    fout.write("   	<td>"+str(info['p0'][8])+"</td>\n")
    fout.write("   	<td>"+str(info['optimisation_bounds'][0][8])+"</td>\n")
    fout.write("   	<td>"+str(info['optimisation_bounds'][1][8])+"</td>\n")
    fout.write("  </tr>\n")
    fout.write("</table>\n")

    for index, sym_set in enumerate(table_values):
        sym = symmetries[index]
        fout.write("<h3>Symmetry: %d  </h3>\n" % (sym))
        fout.write("<table>\n")
        fout.write("<tbody>\n")
        fout.write("   <tr>\n")
        fout.write("     <td><b>Parameter</b></td>\n")
        fout.write("     <td><b>Optimised Value</b></td>\n")
        image_name = "GeometryPlotRotationqalSymmetry"+str(sym)+r"Fold.png"
        fout.write("     <td rowspan=\"0\"><IMG SRC=*** ALT=\"Plot of Model "
                   "Geometry.\" height=100% width=100%>\n</td>\n".replace("***", image_name))
        fout.write("   </tr>\n")
        for count, element in enumerate(sym_set):
            parameter_str = element[2]
            uncertainty_str = element[3]
            fout.write('<tr> <td> %d </td> <td> %s +/- %s </td> '
                       '</tr>\n' % (count+1, parameter_str, uncertainty_str))
        fout.write("</tbody>\n")
        fout.write("</table>\n\n")

    write_html_report_end(fout)



def write_rel_pos_html_report(info):
    """Creates and saves a html report. Contains images of graphs.

    Args:
        dims (int): The dimensions of the data ie 2D or 3D.
        info (dict): A python dictionary containing a collection of useful parameters
            such as the filenames and paths.

    Returns:
       Nothing is returned.
    """


    html_outfile_name = (info['results_dir']+r"/"+info['in_file_no_extension']
                         +r"_"+info['prog']+r"_report.html")

    if info['short_names'] is True:
        html_outfile_name = (info['short_results_dir']+r"/"
                             +info['short_filename_without_extension']+r"_"
                             +info['prog_short_name']+r"_report.html")

    try:
        fout = open(html_outfile_name, "w")
    except IOError as e:
        print("open() failed", e)
        print("The html report file could not be opened for writing. Try using"
              " the s flag to use shorter names.")
        sys.exit()


    fout = write_html_report_start(fout, info)

    report_info = (r"<p>This program ran at "+info['start']+r" on the "
                   +info['host']+r" host system and the data file analysed was "
                   +info['in_file_and_path']+r" which read in "+str(info['values'])
                   +" localisations with "+str(info['columns'])+" columns. ")
    if info['colours_analysed'] is not None:
        report_info = report_info[0:-2] + (', including '
            +repr(len(info['unique_colour_values']))+ ' channels ('
            +np.array2string(info['unique_colour_values'], separator=', ')+ '). ')
    report_info = report_info + ("For the analysis, "
                   +str(info['dims'])+" dimensions were selected, and the filter "
                   "distance was set to "+str(info['filter_dist'])+" nm. ")
    if info['colours_analysed'] == 1:
        report_info = report_info + ('Relative positions were found between localisations '
            'in channel [' +repr(info['start_channel'])+ ']. ')
    if info['colours_analysed'] == 2:
        report_info = report_info + ('Relative positions were found from localisations '
            'in channel [' +repr(info['start_channel'])+ '] to localisations in channel ['
            +repr(info['end_channel'])+ ']. ')
    report_info = report_info + ("This report provides "
                   "images and information on experimental fluorescence super resolution "
                   "light microscopy data.</p>\n")
    fout.write(report_info)
    fout.write("<p></p>\n")

    fout.write("<h2>Plots of Raw Experimental Data</h2>\n")
    image_line = ("<IMG SRC=*** ALT=\"Scatter plot of localisations in +++ "
                  "obtained from experimental fluorescence "
                  "localisation microscopy.\" height=100% width=100%>\n")

    line_dim_0 = image_line.replace("+++", "XY")
    fout.write(line_dim_0.replace("***", "scatter_plot_xy_localisations.png"))

    filename_zoom_scatter_xy = r'scatter_plot_xy_localisations_x'+str(info['zoom'])+'.png'

    image_zoom_line = ("<IMG SRC=*** ALT=\"Scatter plot of localisations in XY obtained "
                       "from experimental fluorescence localisation microscopy. "
                       "This is a zoom of the XY-"
                       "central zone of the data.\" height=100% width=100%>\n")
    fout.write(image_zoom_line.replace("***", filename_zoom_scatter_xy))


    if info['dims'] == 3:
        line_dim_1 = image_line.replace("+++", "XZ")
        fout.write(line_dim_1.replace("***", "scatter_plot_xz_localisations.png"))
        line_dim_2 = image_line.replace("+++", "YZ")
        fout.write(line_dim_2.replace("***", "scatter_plot_yz_localisations.png"))


    fout.write("<h2>Histograms of Analysed Experimental Data</h2>\n")

    line = ("<IMG SRC=*** ALT=\"Histogram of +++ separations between observations "
            "in experimental fluorescence localisation microscopy.\" "
            "height=100% width=100%>\n")

    line_xx = line.replace("+++", "X")
    fout.write(line_xx.replace("***", "histogram_x_separation_in_nm.png"))
    line_yy = line.replace("+++", "Y")
    fout.write(line_yy.replace("***", "histogram_y_separation_in_nm.png"))
    if info['dims'] == 3:
        line_zz = line.replace("+++", "Z")
        fout.write(line_zz.replace("***", "histogram_z_separation_in_nm.png"))



    line_dim_0 = line.replace("+++", "XY")
    fout.write(line_dim_0.replace("***", "histogram_xy_separation_in_nm.png"))
    # And standardised xy separations for 2D report
    if info['dims'] == 2:
        line_dim_1 = line_dim_0.replace("H", "Standardised h")
        fout.write(line_dim_1.replace("***", "histogram_xy_separation_in_nm_2d_standardised.png"))
    if info['dims'] == 3:
        line_dim_1 = line.replace("+++", "XZ")
        fout.write(line_dim_1.replace("***", "histogram_xz_separation_in_nm.png"))
        line_dim_2 = line.replace("+++", "YZ")
        fout.write(line_dim_2.replace("***", "histogram_yz_separation_in_nm.png"))
        line_dim_3 = line.replace("+++", "XYZ")
        fout.write(line_dim_3.replace("***", "histogram_xyz_separation_in_nm.png"))



    write_html_report_end(fout)
