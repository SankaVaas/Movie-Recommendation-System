$(document).ready(function () {
    $('#recommendation-form').submit(function (event) {
        event.preventDefault();
        
        // Get the movie title from the input field
        var movieTitle = $('#movie-title').val();

        // Make an AJAX request to get recommendations
        $.ajax({
            type: 'POST',
            url: '/get_recommendations',
            data: { 'movie_title': movieTitle },
            success: function (data) {
                // Display recommendations on the HTML page
                var recommendations = data['recommendations'];
                var recommendationList = $('<ul>');
                for (var i = 0; i < recommendations.length; i++) {
                    recommendationList.append($('<li>').text(recommendations[i]));
                }
                $('#recommendations').html(recommendationList);
            }
        });
    });
});
