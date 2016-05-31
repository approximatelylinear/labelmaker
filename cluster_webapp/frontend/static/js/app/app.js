
var ENTER_KEY = 13;

var app = app || {};
// app.localStorage = new Backbone.LocalStorage('clusters')


app.explain = false;

app.startExplanation = function () {
	introJs().start();
	app.explain = true;
};


app.Stats = Backbone.Model.extend({
	defaults: {
		total		: 0,
		remaining	: 0,
		completed	: 0
	}
});


//	MODELS
app.Cluster = Backbone.Model.extend({
	defaults	: {
		name			: '',
		tags			: [],
		tagChoices		: [],
		numClusters		:	2,
		data			: null,
		size 			: 0,
		source			: '',		//	File system or database
		sourcename		: '',		//	File name or database name
		group_path		: '',
		table_name		: '',
		db_name			: '',
		h5fname			: '',
		infoFeats 		: [],
		deleted			: false,
		finished		: false,
		visible			: true,
		refresh			: false
	},
	// url			: '/cluster',
	// urlRoot		: '/cluster',
	getUrl		: function () {
		return this.collection.url + this.id;
	},
	children	: null,	//	TODO:	circular dependency? new app.ClusterList(),
	stats		: function () {
		//	TODO:	Sum stats from children.
		if (this.get('children')) {
			return children.stats;
		}
		else { return null; }
	},
	close		: function () {
		//	TODO:	Use `this.destroy`
		this.save({'deleted': true});
	},
	recurse		: function (func, stop) {
		//	Recurse over this model's children
		stop = stop || 10;
		//	Copy the children to a new variable.
		_.each(this.children, function (c) {children.push(c);});
		var stop_idx = 0;
		while (children.length > 0) {
			var child = children.pop();
			//	Apply the specified function to the child.
			func(child);
			//	Refresh the child's tag data.
			child.set('tags', tag_data[child.id]);
			//	Add the next batch of children.
			_.each(child.children, function (c) {children.push(c);});
			if (stop_idx > stop) {
				break;
			}
			stop_idx += 1;
		}
	},
	data_cf		: function () {
		return crossfilter(this.get('data'));
	},
	finish		: function () {
		///////
		return
		///////
		// var that = this;
		// var children = [];
		// //	Copy the children to a new variable.
		// _.each(this.children, function (c) {children.push(c);});
		// var stop_idx = 0;
		// while (children.length > 0) {
		// 	var child = children.pop();
		// 	// child.save({'finished': true});
		// 	//	Add the next batch of children.
		// 	_.each(child.children, function (c) {children.push(c);});
		// 	if (stop_idx > stop) {
		// 		break;
		// 	}
		// 	stop_idx += 1;
		// }
	},
	updateTags 	: function (data) {
		var that = this;
		//	Update child tags.
		var tag_data = data['tags'];
		var new_tag_data = {};
		var children = [];
		//	Copy the children to a new variable.
		if (this.children && 'models' in this.children) {
			_.each(this.children.models, function (c) {children.push(c);});
		}
		//	Transform child keys to ids.
		_.each(_.keys(tag_data), function (k) {
			var new_k = [that.get('h5fname'), k].join('/').replace(/\//gi, '-');
			console.log(k + ' --> ' + new_k);
			new_tag_data[new_k] = tag_data[k];
		});
		tag_data = new_tag_data;
		if (!(this.id in tag_data)) {
			//	Try to find the tags belonging to this model's group.
			var group_path = [this.get('h5fname'), this.get('group_path')].join('/').replace(/\//gi, '-');
			////////
			// console.log(group_path);
			////////
			this.set('tags', tag_data[group_path]);
		}
		else {
			this.set('tags', tag_data[this.id]);
		}
		while (children.length > 0) {
			var child = children.pop();
			//	Refresh the child's tag data.
			child.set('tags', tag_data[child.id]);
			//	Add the next batch of children.
			_.each(child.children, function (c) {children.push(c);});
		}
	},
	removeRow	: function (row_idx) {
		/*
		//	TESTING
		//	-------
		data = _.range(10).map( function (v) { return {'row_idx': v};});
		_.each(data, function (d) { console.log(d.row_idx);})
		idx = 3;
		removed = data.splice(idx, 1);
		_.each(data, function (d) { console.log(d.row_idx);})
		_.each(data.slice(idx), function (d, i) {d['row_idx'] = i + idx;});
		_.each(data, function (d) { console.log(d.row_idx);})
		*/
		var data = this.get('data');
		//	Look for a row with a matching row number.
		var row = _.findWhere(data, {'row_idx': parseInt(row_idx, 10)});
		var removed = [];
		if (row !== undefined) {
			//	Get the row's location in the table.
			var idx = data.indexOf(row);
			//	Splice out the element at `idx`.
			removed = data.splice(idx, 1);
			//	Reset table row indices.
			this.sortData(data, 'row_idx');
			_.each(data.slice(idx), function (d, i) {
				d.row_idx -= 1;
			});
			//	Re-set the data value.
			this.set('data', data);
		}
		return {
			removed	: removed,
			data 	: data
		};
	},
	sortData	: function (data, key, direction) {
		key = key || 'id';
		direction = direction || 1;	//	-1 or 1
		data.sort(function (a, b) {
			var a_val = parseInt(a[key], 10),
				b_val = parseInt(b[key], 10);
			if (a_val > b_val ) { return direction * 1; }
			if (a_val < b_val ) { return direction * -1; }
			return 0;
		});
		return data;
	},
	cluster 	: function (parent_data, data) {
		//////	TEST THIS
		//	Replace this model's data with the new data
		this.set('data', parent_data)
		/////
		var dataLen = data.length;
		this.set('numClusters', dataLen);
		var newData = [];
		if (dataLen > 1) {
			this.children = new app.ClusterList();
			///////////////
			// console.log(data)
			///////////////
			_.each(
				data,
				function (d) {
					// var data = d.data;
					// var tags = d.tags;
					// var group_path = d.group_path;
					// var table_name = d.table_name;
					// var h5fname = d.h5fname;
					// var size = d.size;
					if (d.size > 0) {
						//	Sort the cluster data by id.
						d.data = this.sortData(d.data, 'row_idx');
						//	Update the tags to reflect those for this model.
						d.tags = this.get('tags');
						if ('info_feats' in d) {
							//	Update the info features
							d['infoFeats'] = d['info_feats'];
							delete d['info_feats'];
						}
						//	No need to send data to the server here...
						var cluster = this.children.add(d);
						///////////////////
						console.log(cluster);
						///////////////////
					}
				},
				this
			);
			//	Remove the data from this model.
			this.data = [];
		}
	},
	graph 		: function () {
		//	Put data in scatterplot form:
	    //		[ {x: <INT>, y: <INT>, c: <CATEGORY> }]
	    var graphData = [];
	    _.each(this.get('data'), function (d) {
	    	var d2 = {
	    		x 	: d.v0,
	    		y 	: d.v1,
	    		c 	: d.cluster_id
	    	};
	    	////////////
	    	// console.log(d2);
	    	////////////
	    	graphData.push(d2);
	    });
	    var result = {
	    	'data'		: graphData,
	    	'catNum'	: this.get('numClusters')
	    }
	    return result;
	},
	fakeCluster		: function () {
		//	DEBUGGING
		///////////////////////////
		console.log('Clustering!');
		///////////////////////////
		if (this.has('data')) {
			var data = this.get('data');
			//	N-sect the data.
			var dataLen = data.length;
			if (dataLen > 1) {
				var numClusters = this.get('numClusters') || 2;
				var sliceLen = dataLen / numClusters;
				this.children = new app.ClusterList();
				for (var idx = 0; idx < numClusters; idx +=1 ) {
					var start = sliceLen * idx,
						end = sliceLen * (idx + 1),
						chunk = data.slice(start, end);
					if (chunk.length > 0) {
						var cluster = this.children.create({
							data	: chunk,
							tags	: this.get('tags')
						});
					}
				}
			}
		}
	},
	//	TODO:	Maybe look at completed value for children?
	completed	: function () {
		return this.get('deleted') || this.get('finished');
	}
});
app.RootCluster = new app.Cluster();


app.ClusterList = Backbone.Collection.extend({
	model			: app.Cluster,
	url				: '/cluster/',
// 	localStorage	: app.localStorage,
	//	TODO:	Sum stats from children.
	stats			: new app.Stats({
		total		: this.competed + this.remaining,
		remaining	: this.remaining,
		completed	: this.completed
	}),
	completed		: function () {
		return this.filter(function (c) { return c.completed(); });
	},
	remaining		: function () {
		return this.without.apply(this, this.completed());
	},
	comparator		: function (cluster) { return cluster.get('id'); }
});



//	VIEWS
app.AppView = Backbone.View.extend({
	el				: '#cluster-app',
	statsTemplate	: _.template( $('#stats-template').html() ),
	initialize		: function () {
		this.collection = new app.ClusterList();
		this.$fakeData = this.$('#fake-data');
		this.$main = this.$('#main');
		this.$footer = this.$('#footer');
		//	TODO:	Determine specific event to listen to.
// 		this.listenTo(app.RootCluster, 'all', this.addRoot );
		this.render();
	},
	events			: {
		'click #fake-data'			: 'createFakeData',
		'click #load-data'			: 'loadData',
		// 'click #cluster-refresh'	: 'toggleRefresh'
	},
	addRoot			: function (collection) {
		var collection = collection || this.collection;
		var view = new app.ClusterListView({ 'collection': collection });
		var $clusterList = $('#cluster-list');
		$clusterList.append( view.render().el );
		if (app.explain) {
			var $btn = $('<button></button')
				.text('Keep explaining how!')
				.addClass("btn btn-info")
				.click(function () { return introJs().goToStep(5).start(); });
			//	TODO: 	Only run this on the first item in the cluster list.
			$('.run-cluster-intro').append($btn).append($('<br /><br />'));
		}
	},
	render			: function () {
		this.renderSrcNames();
	},
	renderSrcNames 	: function () {
		// Backbone.ajax(
		$.ajax(
			{
				url			: '/load/',
				type		: 'GET',
				success		: function (data, status, jqXHR) {
					// console.dir(data);
					// console.log(status);
					//	Add file names to select element.
					var $select = $('#load-data-select');
					$select.detach();
					_.each(data.fnames, function (fname) {
						var $opt = $('<option></option>')
						$opt.attr('value', fname)
						$opt.html(fname)
						$select.append($opt)
					})
					$( '#load-data-form div' ).prepend($select);
				}
			}
		)
	},
	toggleRefresh	: function () {},
	createFakeData	: function () {
		console.log('Creating fake data.');
		var fake_data = {
			name: 'Cluster 1',
			data: [
				{name: 'foo'},
				{name: 'bar'},
				{name: 'baz'},
				{name: 'fooo'},
				{name: 'baar'},
				{name: 'baaz'}
			]
		}
		this.addRoot(new app.Cluster(fake_data));
	},
	renderDBInfo	: function (db_info) {
		var tmplt = _.template( $('#form-control-template').html() )
		var $dbInfoForm = $('<form id="db-info-form" class="form-horizontal"></form>')
		////////
		var sizeCtxt = {
			label 		: 'Sample Size',
			labelFor	: 'sampleSize',
			inputType	: 'text',
			inputID		: 'sampleSize_filter',
			placeHolder	: '1000',
		};
		var $sampleSize = $(tmplt(sizeCtxt));
		$dbInfoForm.append($sampleSize);
		///////
		var isUnlabeledCtxt = {
			label 		: 'Only unlabeled data',
			labelFor	: 'isUnlabeled',
			inputType	: 'checkbox',
			inputID		: 'isUnlabeled_filter',
			placeHolder	: null,
		};
		var $isUnlabeled = $(tmplt(isUnlabeledCtxt));
		$dbInfoForm.append($isUnlabeled);
		///////
		_.each(db_info.filters, function (d) {
			var humanLbl = d.name.split('_');
			//	Title-case each word
			humanLbl = _.map(humanLbl, function (d) {
				return d[0].toUpperCase() + d.slice(1);
			});
			humanLbl = humanLbl.join(' ');
			var inputID = d.name + '_filter';
			var filterCtxt = {
				label 		: humanLbl,
				inputType	: 'text',
				inputID		: inputID,
				placeHolder	: ''
			};
			var $filter = $(tmplt(filterCtxt));
			var $choices = $('<div class="hidden"></div>');
			var $choicesClose = $('<span class="btn btn-mini btn-danger pull-right">&times;</span>');
			$choicesClose.on('click', function (evt) {
				return $choices.addClass('hidden');
			});
			var $choicesList = $('<ul></ul>');
			_.each(d.choices, function (c) {
				if (_.isObject(c)) {
					var _key = '<span>' + c.key + '</span>';
					var _val = '<span>(' + c.value + ')</span>';
					$choicesList.append(
						$('<li>' + _key + '&nbsp;&nbsp;' + _val + '</li>')
					);
				}
				else {
					$choicesList.append($('<li>' + c + '</li>'));
				}

			});
			$choices.append($choicesClose);
			$choices.append($('<span class="clearfix"></span>'));
			$choices.append($choicesList);
			$filter.append($choices);
			//	Add a click-handler for showing choices
			$filter.find('#' + inputID + '_choices').on('click', function (evt) {
				return $choices.removeClass('hidden');
			});
			$dbInfoForm.append($filter);
		});
		///////
		_.each(db_info.fields, function (d) {
			var humanLbl = d.name.split('_');
			//	Title-case each word
			humanLbl = _.map(humanLbl, function (d) {
				if (d) {return d[0].toUpperCase() + d.slice(1);}
				else {return d};
			});
			humanLbl = humanLbl.join(' ');
			var fieldCtxt = {
				label 		: humanLbl,
				labelFor	: d.name + '_field',
				inputType	: 'checkbox',
				inputID		: d.name + '_field',
				placeHolder	: null
			};
			var $field = $(tmplt(fieldCtxt));
			$dbInfoForm.append($field);
		});
		return $dbInfoForm;
	},
	getDBInfoForm	: function () {
		var formData = {
			filters	: {},
			fields	: {}
		};
		$( '#db-info-form div' ).find( 'input' ).each(
			function (i, elt) {
				var $elt = $(elt);
				var id = elt.id;
				var dataType = id.split('_').slice(-1)[0];
				var key;
				if ( dataType === 'filter' ) {
					key = 'filters';
				}
				else {
					key = 'fields';
				}
				var inputType = $elt.prop('type');
				var val = '';
				if ( inputType === 'text' ) {
					val = $elt.val().trim();
				}
				else if ( inputType === 'checkbox' ) {
					val = $elt.prop('checked');
				}
				formData[key][id] = val;
			}
		);
		if (_.size(formData.filters) > 0 || _.size(formData.fields) > 0) {
			formData.has_user_data = true;
		}
		return formData;
	},
	getLoadDataForm	: function () {
		var formData = {};
		formData['sourcename'] = $('#load-data-select').val();
		$( '#load-data-form div' ).find( 'input' ).each(
			function (i, elt) {
				var $elt = $(elt);
				var val = $elt.val().trim();
				if ( val != '') {
					if ( elt.id === 'cluster-name') {
						formData['name'] = val;
					}
					else if ( elt.id === 'cluster-refresh') {
						formData['refresh'] = $elt.prop('checked');
					}
					else {
						formData[elt.id] = val;
					}
				}
			}
		);
		return formData;
	},
	loadData		: function (evt) {
		evt.preventDefault();
		var loadDataForm = this.getLoadDataForm();
		var dbInfoForm = this.getDBInfoForm();
		var formData = _.extend(loadDataForm, dbInfoForm);
		var that = this;
		if (formData['name'].length > 0) {
			//	Add a loading gif.
			var $clusterList = $('#cluster-list');
			var $loading = $clusterList.children('img.loading');
			if (!$loading.length) {
				$loading = $('<img class="loading" src="../static/img/ajax-loader.gif"></img>');
				$clusterList.append($loading);
			}
			that.$('#load-data').addClass('hidden');
			//	Get data from server
			Backbone.ajax(
				{
					url			: '/load/',
					data 		: JSON.stringify(formData),
					type		: 'POST',
					dataType	: 'json',
					contentType : 'application/json',
					success	: function (data, status, jqXHR) {
						// console.dir(data);
						// console.log(status);
						// console.log(jqXHR);
						that.$('#load-data').removeClass('hidden');
						if (_.has(data, 'db_info')) {
							var db_info = data.db_info;
							//	Remove the loading gif.
							$('#cluster-list').children('img.loading').remove();
							//	Add a form requesting user input
							var $dbInfoForm = that.renderDBInfo(db_info);
							$('#load-data-div').append($dbInfoForm);
							//	Change to sample button
							that.$('#load-data').text('Sample from Database');
						}
						else if (_.has(data, 'children')) {
							_.each(data.children, function (d) {
								var root = that.collection.create(d);
							});
							//	Add the cluster list views in the collection.
							that.addRoot();
							//	Remove the loading gif.
							$('#cluster-list').children('img.loading').remove();
							// that.$('.children').children('img.loading').remove();
							$('#db-info-form').remove();
						}
					},
					error: function( req, status, err ) {
						that.$('#load-data').removeClass('hidden');
						console.log( 'something went wrong', status, err );
					}
				}
			);
			// var ajaxOpts = {
			// 	url			: '/load',
			// 	type		: 'POST',
			// 	success		: function (model, data, jqXHR) {
			// 		// console.log('Loaded data!')
			// 		// console.dir(data);
			// 		// console.log(status);
			// 		//	Remove the loading gif.
			// 		$('#cluster-list').children('img.loading').remove();
			// 		that.addRoot();
			// 	}
			// };
			// var root = this.collection.create(
			// 	formData,
			// 	ajaxOpts
			// );
		}
	},
});



app.ClusterListView = Backbone.View.extend({
	tagName		: 'ul',
	template	: _.template( $('#cluster-list-template').html() ),
	events		: {
		'click .close-list'			: 'close',
		'click .save-list'			: 'finish',
		'click .toggle-list-size'	: 'toggleSize',
	},
	collection	: null,
	initialize	: function () {
		this.childViews = [];
		this.parentView	= null;
	},
	render		: function () {
		this.addAll();
		return this;
	},
	renderTags	: function () {
		////////////
		// console.log(this.childViews);
		///////////
		_.each(this.childViews, function (v) { v.renderTags(); });
	},
	addCluster	: function (cluster) {
		//	Make sure the cluster has the right url:
		cluster.url = this.collection.url + '/' + cluster.id;
		var view = new app.ClusterView({ model: cluster});
		//	Keep parent and child references.
		// view.parentView = this;
		this.childViews.push(view);
		this.$el.append( view.render().el );
	},
	addAll		: function () {
		this.$el.html('');
		//	TODO:	Grab the clusters we need to render.
		if (this.collection) {
			this.collection.each(this.addCluster, this);
		}
	},
	close		: function (evt) {
		evt.preventDefault();
		evt.stopPropagation();
		//	 Delete cluster, perform no further clustering.
		if (this.collection) {
			this.childViews = [];
			this.collection.each(function (cluster) { cluster.close(); });
		}
	},
	finish		: function (evt) {
		evt.preventDefault();
		evt.stopPropagation();
		//	 Save cluster, perform no further clustering.
		if (this.collection) {
			this.collection.each(function (cluster) { cluster.finish(); });
		}
	},
	toggleSize	: function (evt) {
		evt.preventDefault();
		evt.stopPropagation();
		//	Toggle size for each cluster in the list
		this.$el.toggleClass();
	},
	update		: function (evt) {
		evt.preventDefault();
		evt.stopPropagation();
		//	Save each cluster in the list
	},
	hidden				: false,
	toggleSize			: function (evt) {
		evt.preventDefault();
		evt.stopPropagation();
		//	Change cluster visibility
		this.hidden = !this.hidden;
		this.$el.toggleClass( 'hidden', this.hidden );
	},
});



app.ClusterView = Backbone.View.extend({
	tagName				: 'li',
	template			: _.template( $('#cluster-template').html() ),
	scatterpltTemplate	: _.template( $('#scatterplot-template').html() ),
	events				: {
		'click .cluster'			: 'cluster',
		'click .finish'				: 'finish',
		'click .export'				: 'exportData',
		'click .graph'				: 'graphData',
		'click .close_'				: 'close',
		'click .toggle-size'		: 'toggleSize',
		'click label.edit-tags'		: 'editTags',
		'keypress input.edit-tags'	: 'updateTagsOnEnter',
		// 'blur input.edit-tags'		: 'updateTagsOnBlur'
	},
	model				: null,
	initialize			: function () {
		this.childrenView = null;
		this.parentView = null;
		if (this.model) {
			//	TODO: Listen for changes to this models children:
			//		re-render size
			//		re-render download link
			//		remove references to deleted children from the childrenView.

			//this.listenTo(this.model, 'change', this.render());
			//this.listenTo(this.model, 'change:data', this.addData());
			//this.listenTo(this.model, 'change:children', this.addChildren());
			////////////////////////////////////
			// console.log(this.model.get('data'));
			////////////////////////////////////
			if (this.model.children) {
				this.model.children.fetch();
				this.addChildren();
			}
		}
	},
	render				: function () {
		var context = this.model.toJSON();
		context.cid = this.model.cid;
		context.tagstr = this.model.get('tags').join(', ').trim();
		this.$el.html(this.template(context));
		this.renderTags();
		this.renderInfoFeats();
		this.addData();
		this.addChildren();
		if (!this.$tagInput) {
			this.$tagInput = this.$('input.edit-tags').first();
		}
		if (!this.$numClsInput) {
			this.$numClsInput = this.$('.edit-num-cluster');
		}
		return this;
	},
	graphData			: function () {
		var context = {
			'title'	: '',
			'about'	: ''
		};
		var $charts = this.$('>div>div.charts');
		$charts.html(this.scatterpltTemplate(context));
		$('#c4 > div.charts > div.scatterplot')

		var selector = '#' + this.model.cid + '> div.charts > div.scatterplot svg.chart';
		var graphInfo = this.model.graph();
		//	Create the scatterplot and insert it in the DOM.
		ScatterPlot(graphInfo.data,  graphInfo.catNum, selector);
	},
	formatInfoFeats	: function () {
		var infoFeats = this.model.get('infoFeats');
		var newInfoFeats = [];
		_.each(infoFeats, function (d) {
			newInfoFeats.push({
				k 	: d['words'],
				v 	: d['magnitude']
			});
		});
		// this.model.sortData(newInfoFeats, 'v', 1);
		return newInfoFeats;
	},
	renderInfoFeats		: function () {
		var infoFeats = this.formatInfoFeats();
		if (infoFeats.length > 0) {
			var $infoFeats = this.$('> div > div.info-feats');
			// var $parent = $infoFeats.parent();
			var $data = this.$('.data');
			var infoFeatsLen = infoFeats.length;
			$infoFeats.detach();
			$infoFeats.append($('<hr />'));
			$infoFeats.append($('<span>Characteristic Words:&nbsp;&nbsp;</span>'))
			//	Render colors
			_.each(infoFeats, function (d, i) {
				var feat = d.k;
				if (i < infoFeatsLen) { feat += ' '; }
				// var c = d3.rgb('#cccccc');
				// c = c.darker(d.v);
				var c = d3.hsl(130, .5, Math.max(.85 - d.v, .15));
				//////////////
				console.log('[' + i + '] ' + d.k + ' : ' + c.toString());
				//////////////
				var $elem = $('<span></span')
					.text(feat)
					.css('color', c.toString());
				/////////
				// console.log($elem)
				/////////
				$infoFeats.append($elem)
			});
			$infoFeats.append($('<hr />'));
			// $parent.append($infoFeats);
			$infoFeats.insertBefore($data);
		}
	},
	addData				: function () {
		//	Add a <tr> elem for each data point in this cluster.
		var data = this.model.get('data');
		//	Sort the data by row index.
		data = this.model.sortData(data, 'row_idx');
		//	Get a reference to the table DOM element and zero-out its data.
		var $data = this.$('.data').html('');
		var $body = $('<tbody></tbody>');
		//	Add a caption describing the table contents.
		$data.append($('<caption></caption>').text('Cluster ' + this.model.id));
		//	$data.append(this.model.data_cf());
		//	Add header
		var $header = null;
		var that = this;
		var displayCols = [
			'clean_content', 'cluster_id',
			'parent_id', 'row_idx',
			'conversation', 'date'
		]
		_.each(
			data,
			function (d) {
				//	Display only these columns.
				d = _.pick(d, displayCols);
				if (!$header) {
					$header = $('<thead></thead>');
					var $row = $('<tr class=".data-header"></tr>');
					for (k in d) {
						var $elem = $('<th>' + k + '</th>');
						$row.append($elem);
					}
					$header.append($row);
					$data.append($header);
				}
				//	Combine the cluster id and the row id
				var row_idx = that.model.id + '|' + d.row_idx;
				var $row = $('<tr class=".data-body"></tr>').attr('id', row_idx);
				//	Add a click-handler for removing the row.
				$row.on('dblclick', function (evt) { return that.removeData(evt, that) } );
				for (k in d) {
					var $elem = $('<td>' + d[k] + '</td>');
					$row.append($elem);
				}
				$body.append($row);
			},
			this
		);
		$data.append($body);

	},
	removeData			: function (evt, view) {
		/////
		// console.log('Deleting data!');
		/////
		evt.preventDefault();
		evt.stopPropagation();
		var $target = $(evt.currentTarget);
		var row_idx = evt.currentTarget.id.split('|')[1];
		var model = view.model;
		var url = model.getUrl();
		var result = model.removeRow(row_idx);
		var data = JSON.stringify({
			h5fname		: model.get('h5fname'),
			group_path	: model.get('group_path'),
			table_name	: model.get('table_name'),
			idx 		: row_idx
		});
		var that = view;
		Backbone.ajax(
			{
				url			: url + '/remove_row/',
				data 		: data,
				type		: 'POST',
				dataType	: 'json',
				contentType : 'application/json',
				success	: function (data, status, jqXHR) {
					///////////////////
					// console.log(status);
					// console.log(data);
					///////////////////
					$target.off();	//	Remove all events to prevent memory leakage.
					$target.addClass('error').fadeOut( 500, function() { this.remove(); });
					//	Re-render the rows.
					that.addData();
					//	Adjust the cluster size
					var $clusterSize = that.$(".cluster-size strong");
					//////
					console.log($clusterSize);
					/////
					var prev = parseInt($clusterSize.text(), 10);
					$clusterSize.text(prev - 1);
				},
				error: function( req, status, err ) {
					console.log( 'something went wrong', status, err );
				}
			}
		);
	},
	exportData	: function (evt) {
		evt.preventDefault();
		evt.stopPropagation();
		// var $exportBtn = this.$('.export');
		var $exportBtn = this.$('.export').first();
		var that = this;
		var url = this.model.getUrl();
		///////////////////////////
		console.log('url for model #' + this.model.id + ': ' + url);
		///////////////////////////
		//	Add a loading gif.
		$exportBtn.append($('<img class="loading" src="../static/img/ajax-loader.gif"></img>'));
		//	Get data from server
		Backbone.ajax(
			{
				url			: url + '/export/',
				data 		: JSON.stringify({
					h5fname		: this.model.get('h5fname'),
	                group_path	: this.model.get('group_path'),
	                db_name		: this.model.get('db_name')
				}),
				type		: 'POST',
				dataType	: 'json',
				contentType : 'application/json',
				success	: function (data, status, jqXHR) {
					if ((data.error !== undefined) && (data.error === 1)) {
						////////////////////////////
						console.log('something went wrong', status, data.msg );
						////////////////////////////
						that.$el.addClass('error');
						that.$el.prepend($('<span>Failed to export this cluster.</span>'));
					}
					else {
						//////////////////
						console.log('Exported data on the server!')
						console.dir(data);
						console.log(status);
						//////////////////
						//	Append link to file
						// var href = data.fname;
						var href = '/cluster/' + data.fname + '/download/';
						var $exportLink = $('<a><i class="icon-download-alt icon-white"></i> Download</a>').attr({
							'class'		: "download btn btn-success",
							'target'	: '_blank',
							'href'		: href
						});
						$exportBtn.replaceWith($exportLink);
						//	Remove the loading gif.
						// $exportBtn.children('img.loading').remove();
					}
				},
				error: function( req, status, err ) {
					////////////////////////////
					console.log('something went wrong', status, err);
					////////////////////////////
				}
			}
		);
	},
	addChildren			: function () {
		//	Call the `render` method for each child
		var $children = this.$('.children').html('');
		if (this.model.children) {
			var childrenView = new app.ClusterListView(
				{ collection: this.model.children }
			);
			//	Keep a reference to the view so that we can remove it later.
			// childrenView.parentView = this;
			this.childrenView = childrenView;
			$children = $children.html(childrenView.render().el);
		}
	},
	renderTags			: function () {
		var that = this;
		var $tagDisplay = this.$('label.edit-tags').first();
		var $tagInput = this.$('input.edit-tags').first();
		var tags = this.model.get('tags').join(', ').trim();
		if (tags.length > 0) {
			$tagDisplay.text(tags);
			$tagInput.prop('value', tags);
		}
		else {
			$tagDisplay.html($('<em>Click to add tags</em>'));
		}
		/*
		var tagElems = [];
		_.each(tags, function (d) {
			var $tagElem = $('<span></span')
				.text(d)
				//	Add a click-handler for removing the tag.
				.on('dblclick', function (evt) { return that.removeTag(evt, that) } );
			tagElems.append($tagElem);
		});
		*/
		if (this.childrenView) {
			this.childrenView.renderTags();
		}
	},
	removeTag			: function (evt, view) {
		/////
		// console.log('Deleting data!');
		/////
		// evt.preventDefault();
		// evt.stopPropagation();
		// var $target = $(evt.currentTarget);
		// var row_idx = evt.currentTarget.id.split('|')[1];
		// var model = view.model;
		// var url = model.getUrl();
		// var result = model.removeRow(row_idx);
		// var data = JSON.stringify({
		// 	h5fname		: model.get('h5fname'),
		// 	group_path	: model.get('group_path'),
		// 	table_name	: model.get('table_name'),
		// 	idx 		: row_idx
		// });
		// var that = view;
		// Backbone.ajax(
		// 	{
		// 		url			: url + '/remove_row',
		// 		data 		: data,
		// 		type		: 'POST',
		// 		dataType	: 'json',
		// 		contentType : 'application/json',
		// 		success	: function (data, status, jqXHR) {
		// 			///////////////////
		// 			// console.log(status);
		// 			// console.log(data);
		// 			///////////////////
		// 			$target.off();	//	Remove all events to prevent memory leakage.
		// 			$target.addClass('error').fadeOut( 500, function() { this.remove(); });
		// 			//	Re-render the rows.
		// 			that.addData();
		// 		},
		// 		error: function( req, status, err ) {
		// 			console.log( 'something went wrong', status, err );
		// 		}
		// 	}
		// );
	},
	cluster				: function (evt) {
		evt.preventDefault();
		evt.stopPropagation();
		var that = this;
		var url = this.model.getUrl();
		var numClusters = this.$numClsInput.val() || 2;
		this.model.set('numClusters', numClusters);
		///////////////////////////
		console.log('url for model #' + this.model.id + ': ' + url);
		///////////////////////////
		//	Add a loading gif.
		var $clusterBtn = this.$('.cluster').first();
		$clusterBtn.append($('<img class="loading" src="../static/img/ajax-loader.gif"></img>'));
		//	Get data from server
		Backbone.ajax(
			{
				url			: url + '/cluster/',
				data 		: JSON.stringify(this.model.toJSON()),
				type		: 'POST',
				dataType	: 'json',
				contentType : 'application/json',
				success	: function (data, status, jqXHR) {
					// console.log('Created a new cluster on the server!')
					// console.dir(data);
					// console.log(status);
					that.model.cluster(data.parent, data.children);
					that.addChildren();
					var $elem = that.$('#' + that.model.cid);
					$elem.children('.data').remove();
					$elem.children('.cluster').remove();
					//	Disable the graph button for the node that was clustered.
					that.$('>div>div>div>button.cluster').prop('disabled', true);
					//	Enable the graph button for the node that was clustered.
					that.$('>div>div>div>button.graph').prop('disabled', false);
					//	Remove the loading gif.
					// that.$('.children').children('img.loading').remove();
					$clusterBtn.children('img.loading').remove();
				},
				error: function( req, status, err ) {
					console.log( 'something went wrong', status, err );
				}
			}
		);
	},
	hidden				: false,
	toggleSize			: function (evt) {
		evt.preventDefault();
		evt.stopPropagation();
		//	Change cluster visibility
		this.hidden = !this.hidden;
		this.$('.data').toggleClass( 'hidden', this.hidden );
		this.$('.children').toggleClass( 'hidden', this.hidden );
		this.$('.toggle-size').text(this.hidden ? '+' : '-');
	},
	editTags			: function (evt) {
		evt.preventDefault();
		evt.stopPropagation();
		//	Switch to editing view
		this.$el.addClass('editing');
		this.$tagInput.focus();
		// this.$tagInput.addClass('editing');
	},
	updateTags 			: function (context) {
		var $tagInput = context.$('input.edit-tags');
		var value = $tagInput.val().trim();
		var values = _.map(value.split(','), function (s) {return s.trim().toLowerCase()});
		if (values.length > 0 && values !== context.model.tags) {
			var that = context;
			var $tagDisplay = context.$('label.edit-tags');
			//	Add a loading gif.
			var $loading = $tagDisplay.parent().find('img.loading');
			if (!$loading.length) {
				$loading = $('<img class="loading" src="../static/img/ajax-loader.gif"></img>');
				$loading = $loading.insertBefore($tagDisplay);
			}
			// else {
				// $loading = $loading[0];
			// }
			//////////////////////////////////////////////////////
			console.log(value + ' ===> ');
			_.each(values, function (v) { console.log('    ' + v); });
			/////////////////////////////////////////////////////
			context.model.save(
				{
					tags 	: values,
					db_name	: this.model.get('db_name')
				},
				{ 	url 	: context.model.getUrl() + '/',
					success	: function (model, data, jqXHR) {
						//////////
						// console.log(data)
						//////////
						that.model.updateTags(data);
						//	Re-render this node's tags and its children.
						that.renderTags();
						//	Remove the loading gif.
						$loading.remove();
						var $success = $('<span><img height=25 width=25 class="success" src="../static/img/success.png"></img></span>');
						$success = $success.insertBefore($tagDisplay);
						$success.fadeOut( 300, function() {$(this).remove();});
						that.$el.removeClass('editing');
					}
				}
			);
		}
	},
	updateTagsOnBlur	: function (evt) {
		evt.stopPropagation();
		// this.updateTags(this);
	},
	updateTagsOnEnter	: function (evt) {
		evt.stopPropagation();
		if (evt.which === ENTER_KEY) {
			this.updateTags(this);
			// var $tagInput = this.$('input.edit-tags');
			// $tagInput.trigger('blur');
		}
	},
	editNumCls			: function (evt) {
		evt.preventDefault();
		evt.stopPropagation();
		//	Switch to editing view
		this.$el.addClass('editing');
		this.$numClsInput.focus();
	},
	updateNumClsOnEnter	: function (evt) {
		evt.stopPropagation();
		if (evt.which === ENTER_KEY) {
			var value = parseInt(this.$numClsInput.val().trim(), 10);
			if (value) {
				this.model.numClusters = value;
			}
		}
	},
	finish				: function (evt) {
		evt.preventDefault();
		evt.stopPropagation();
		//	Save to database, perform no further clustering.
		// this.model.finish();
		this.remove();
	},
	close				: function (evt) {
		evt.preventDefault();
		evt.stopPropagation();
		var that = this;
		//	 Delete cluster, perform no further clustering.
		this.model.destroy({
			url: 		this.model.getUrl() + '/',
			success	: 	function (model, response) {
				///////////
				// console.log(model);
				// console.log(response);
				////////////
				if ((response.error !== undefined) && (response.error === 1)) {
					that.$el.addClass('error');
					that.$el.prepend($('<span>Failed to delete this cluster.</span>'));
				}
				else {
					// if (this.parentView) {
					// 	this.parentView.childViews();
					// }
					// var views = [this.childrenView];
					// var stop_idx = 0;
					// while (views.length > 0) {
					// 	var view = views.pop();
					// 	//	Add the next batch of children.
					// 	while (child.children.length > 0) {
					// 		children.push(child.children.pop());
					// 	}
					// 	//	Remove the child from the DOM and memory.
					// 	child.remove();
					// 	if (stop_idx > stop) {
					// 		break;
					// 	}
					// 	stop_idx += 1;
					// }
					that.childrenView = null;
					that.remove();
				}
			},
			error	: 	function (model, jqXHR) {
				///////////
				console.log(model);
				console.log(jqXHR);
				////////////
				that.$el.adClass('error');
				that.$el.prepend($('<span>Failed to delete this cluster</span>'));
			}
		});
	}
});


app.StatsView = Backbone.View.extend({});



app.App = new app.AppView();

