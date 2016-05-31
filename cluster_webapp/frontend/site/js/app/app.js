
var ENTER_KEY = 13;

var app = app || {};
app.localStorage = new Backbone.LocalStorage('clusters')


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
		tags			: '',
		numClusters		:	2,
		data			: null,
		deleted			: false,
		finished		: false,
		visible			: true
	},
	url			: '',	//	/clusters
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
	finish		: function () {
		this.save({'finished': true});
	},
	data_cf		: function () {
		return crossfilter(this.get('data'));
	},
	cluster		: function () {
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
	url				: '/clusters',
	localStorage	: app.localStorage,
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
	},
	events			: {
		'click #fake-data'	: 'createFakeData',
		'click #load-data'	: 'loadData'
	},
	addRoot			: function (cluster) {
		var view = new app.ClusterView({ model: cluster });
		$('#cluster-list').append( view.render().el );
	},
	render			: function () {},
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
	loadData	: function (evt) {
		evt.preventDefault();
		var formData = {};
		$( '#load-data div' ).children( 'input' ).each(
			function (i, elt) {
				var val = $(elt).val().trim()
				if ( val != '') {
					formData[elt.id] = val;
				}
			}
		);
		this.collection.create(
			formData,
			{
				url		: this.collection.url + '/load',
				success	: function (data, status, jqXHR) {
					//console.log('Loaded data and created a new cluster!')
					console.dir(data);
					console.log(status);
				}
			}
		);
		//	TODO:	add data to the root object.
		//	this.addRoot(new app.Cluster(formData));
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
	},
	render		: function () {
		this.addAll();
		return this;
	},
	addCluster	: function (cluster) {
		var view = new app.ClusterView({ model: cluster});
		//	TODO:	Will this preserve a copy of the view so that it can continue
		//			to listen for changes and events?
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
	events				: {
		'click .cluster'		: 'cluster',
		'click .save'			: 'finish',
		'click .close'			: 'close',
		'click .toggle-size'	: 'toggleSize',
		'click .edit-tags'		: 'editTags',
		'keypress .edit-tags'	: 'updateTagsOnEnter',
	},
	model				: null,
	initialize			: function () {
		if (this.model) {
			//this.listenTo(this.model, 'change', this.render());
			//this.listenTo(this.model, 'change:data', this.addData());
			//this.listenTo(this.model, 'change:children', this.addChildren());
			////////////////////////////////////
			console.log(this.model.get('data'));
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
		this.$el.html(this.template(context));
		this.addData();
		this.addChildren();
		if (!this.$tagInput) {
			this.$tagInput = this.$('.edit-tags');
		}
		if (!this.$numClsInput) {
			this.$numClsInput = this.$('.edit-num-cluster');
		}
		return this;
	},
	addData				: function () {
		//	Add a <li> item for each item in the data for this cluster.
		var data = this.model.get('data');
		var $data = this.$('.data').html('');	//	Table elem.
		var $body = $('<tbody></tbody>');
		//	$data.append(this.model.data_cf());
		//	Add header
		var $header = null;
		_.each(
			data,
			function (d) {
				if (!$header) {
					$header = $('<thead></thead>');
					var $row = $('<tr></tr>');
					for (k in d) {
						var $elem = $('<th>' + k + '</th>');
						$row.append($elem);
					}
					$header.append($row);
					$data.append($header);
				}
				var $row = $('<tr></tr>');
				for (k in d) {
					var $elem = $('<td>' + d[k] + '</td>');
					$row.append($elem);
				}
				//	$body = $body.append($row);
				$body.append($row);
				///////////////////
				//console.dir($elem)
				//console.dir($data)
				///////////////////
			},
			this
		);
		$data.append($body);
	},
	addChildren			: function () {
		//	Call the `render` method for each child
		var $children = this.$('.children').html('');
		if (this.model.children) {
			var childrenView = new app.ClusterListView(
				{ collection: this.model.children }
			);
			$children = $children.html(childrenView.render().el);
		}
	},
	/*
	dataList			: function (div) {
		var that = this;
		var data = this.model.get('data');
		div.each(function () {
			var datum = d3.select(that.$('.data'))
				.selectAll('.item')
				.data(data, function (d) {return d.name; });
			datum.enter().append('div')
				.attr('class', 'name')
			datum.blah()
		});
	},
	*/
	cluster				: function (evt) {
		evt.preventDefault();
		evt.stopPropagation();
		this.model.cluster();
		////////////////////////////////
		//	console.dir(this.model.children);
		////////////////////////////////
		this.addChildren();
	},
	hidden				: false,
	toggleSize			: function (evt) {
		evt.preventDefault();
		evt.stopPropagation();
		//	Change cluster visibility
		this.hidden = !this.hidden;
		this.$('.data').toggleClass( 'hidden', this.hidden );
		this.$('.children').toggleClass( 'hidden', this.hidden );
	},
	editTags			: function (evt) {
		evt.preventDefault();
		evt.stopPropagation();
		//	Switch to editing view
		this.$el.addClass('editing');
		this.$tagInput.focus();
	},
	updateTagsOnEnter	: function (evt) {
		evt.stopPropagation();
		if (evt.which === ENTER_KEY) {
			var value = this.$tagInput.val().trim();
			//	TODO:	Split tags and store each one as a list item.
			if (value) {
				this.model.save({ tags: value});
			}
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
			//	TODO:	Split tags and store each one as a list item.
			if (value) {
				this.model.save({ numCluster: value});
			}
		}
	},
	finish				: function (evt) {
		evt.preventDefault();
		evt.stopPropagation();
		//	Save to database, perform no further clustering.
		this.model.finish();
		this.remove();
	},
	close				: function (evt) {
		evt.preventDefault();
		evt.stopPropagation();
		//	 Delete cluster, perform no further clustering.
		this.model.close();
		this.remove();
	}
});


app.StatsView = Backbone.View.extend({});



app.App = new app.AppView();

